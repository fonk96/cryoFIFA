#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch cryofifa.py for safer single-node DDP/NCCL behavior.
Usage:
  python patch_cryofifa_ddpfix_v2.py cryofifa.py cryofifa_ddpfix.py
"""
import re
import sys
from pathlib import Path


def die(msg):
    raise SystemExit("ERROR: " + msg)


def add_import_datetime(text: str) -> str:
    if re.search(r"^import\s+datetime\s*$", text, re.M):
        return text
    if re.search(r"^import\s+time\s*$", text, re.M):
        return re.sub(r"^(import\s+time\s*)$", r"\1\nimport datetime", text, count=1, flags=re.M)
    return "import datetime\n" + text


HELPERS = r'''

def get_global_min_free_mem_gb(device):
  """Query only the current rank's GPU, then all-reduce the global minimum.
  Avoid torch.cuda.mem_get_info(gpui) over all GPUs; that creates CUDA contexts
  on every GPU from every rank and can poison NCCL/DDP behavior.
  """
  free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
  t = torch.tensor([free_bytes / 1024**3], device=device, dtype=torch.float32)
  distr.all_reduce(t, op=distr.ReduceOp.MIN)
  return float(t.item())


def ddp_wrap(module, local_rank, args):
  return DDP(
    module,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=False,
    gradient_as_bucket_view=args.fast_ddp,
  )
'''


def add_helpers(text: str) -> str:
    if "def get_global_min_free_mem_gb" in text and "def ddp_wrap" in text:
        return text
    m = re.search(r"^def\s+main\s*\(\s*\)\s*:\s*$", text, re.M)
    if not m:
        die("cannot find def main():")
    return text[:m.start()] + HELPERS + "\n" + text[m.start():]


NEW_SETUP = '''  local_rank = int(os.environ["LOCAL_RANK"])
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  args.world_size = world_size

  setup_for_distributed(rank == 0)

  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")

  distr.init_process_group(
    backend="nccl",
    init_method="env://",
  )

  # CPU-side barrier group for long rendering / checkpoint sections.
  # Other ranks wait here on CPU instead of entering NCCL collectives while one rank renders/writes maps.
  sync_pg = distr.new_group(
    backend="gloo",
    timeout=datetime.timedelta(hours=12),
  )'''


def replace_setup(text: str) -> str:
    # If already patched, do nothing.
    if "sync_pg = distr.new_group" in text and "rank = int(os.environ[\"RANK\"])" in text:
        return text
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*local_rank\s*=\s*int\s*\(\s*os\.environ\s*\[\s*['\"]LOCAL_RANK['\"]\s*\]\s*\)\s*$", line):
            start = i
            break
    if start is None:
        die("cannot find local_rank = int(os.environ['LOCAL_RANK']) block")
    end = None
    for j in range(start, min(start + 20, len(lines))):
        if re.match(r"^\s*device\s*=\s*torch\.device\s*\(\s*['\"]cuda['\"]\s*\)\s*$", lines[j]):
            end = j
            break
    if end is None:
        die("found local_rank but cannot find following device=torch.device('cuda')")
    new_lines = lines[:start] + NEW_SETUP.splitlines() + lines[end+1:]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def patch_ddp_calls(text: str) -> str:
    # Convert simple one-line DDP assignments to ddp_wrap.
    out = []
    changed = 0
    pat = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*=\s*DDP\s*\(\s*([A-Za-z_]\w*)\s*,.*device_ids\s*=\s*\[\s*local_rank\s*\].*\)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            indent, lhs, arg0 = m.groups()
            out.append(f"{indent}{lhs} = ddp_wrap({arg0}, local_rank, args)")
            changed += 1
        else:
            out.append(line)
    text2 = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if changed == 0 and "ddp_wrap(" not in text2:
        die("did not patch any DDP(...) calls; DDP calls may be multi-line/unexpected")
    return text2


def patch_memory_queries(text: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0
    changed = 0
    while i < len(lines):
        line = lines[i]
        # Initial block: free_gpus_memorys=np.zeros(...) ... current_free_mem=min(...)
        if re.match(r"^\s*free_gpus_memorys\s*=\s*np\.zeros\s*\(\s*args\.world_size\s*\)\s*$", line):
            indent = re.match(r"^(\s*)", line).group(1)
            j = i + 1
            found = False
            while j < len(lines) and j < i + 12:
                if re.match(r"^\s*current_free_mem\s*=\s*min\s*\(\s*free_gpus_memorys\s*\)\s*$", lines[j]):
                    found = True
                    break
                j += 1
            if found:
                out.append(f"{indent}current_free_mem = get_global_min_free_mem_gb(device)")
                i = j + 1
                changed += 1
                continue
        # Epoch-end block: for gpui in range(args.world_size): ... mem_get_info(gpui) ... current_free_mem=min(...)
        if re.match(r"^\s*for\s+gpui\s+in\s+range\s*\(\s*args\.world_size\s*\)\s*:\s*$", line):
            indent = re.match(r"^(\s*)", line).group(1)
            window = "\n".join(lines[i:i+8])
            if "torch.cuda.mem_get_info(gpui)" in window and "current_free_mem" in window:
                j = i + 1
                found = False
                while j < len(lines) and j < i + 12:
                    if re.match(r"^\s*current_free_mem\s*=\s*min\s*\(\s*free_gpus_memorys\s*\)\s*$", lines[j]):
                        found = True
                        break
                    j += 1
                if found:
                    out.append(f"{indent}current_free_mem = get_global_min_free_mem_gb(device)")
                    i = j + 1
                    changed += 1
                    continue
        out.append(line)
        i += 1
    text2 = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if "torch.cuda.mem_get_info(gpui)" in text2:
        die("still found torch.cuda.mem_get_info(gpui); memory-query patch incomplete")
    if changed == 0 and "get_global_min_free_mem_gb(device)" not in text2:
        die("did not patch memory-query blocks")
    return text2


def patch_save_pose(text: str) -> str:
    # Make pose output per-rank to avoid multi-rank append to same file.
    text2 = re.sub(
        r"(out_pose_name\s*=\s*args\.render_path\s*\+\s*'//'\s*\+\s*args\.output_pose_file\s*\+\s*'_'\s*\+\s*str\(epoch\+1\)\s*\+\s*)'\.star'",
        r"\1f'_rank{rank}.star'",
        text,
    )
    return text2


def patch_render_block(text: str) -> str:
    if "distr.barrier(group=sync_pg)" not in text:
        # Replace the save interval + writer if line.
        old_pat = re.compile(r"^(\s*)if\s*\(epoch\+1\)%args\.save_interval==0\s+and\s+local_rank==args\.write_device\s*:\s*$", re.M)
        m = old_pat.search(text)
        if not m:
            die("cannot find epoch-end render/save if line")
        indent = m.group(1)
        repl = (
            f"{indent}if (epoch+1)%args.save_interval==0:\n"
            f"{indent}  distr.barrier(group=sync_pg)\n\n"
            f"{indent}if (epoch+1)%args.save_interval==0 and rank==args.write_device :"
        )
        text = text[:m.start()] + repl + text[m.end():]
    # Insert _bare render models after y=None in the writer block, if not already present.
    if "_render_efe = _bare(Encoder_Feature_Extractor)" not in text:
        old = "      y=None\n      pred_images_3d=None\n"
        new = "      y=None\n      pred_images_3d=None\n\n      _render_efe = _bare(Encoder_Feature_Extractor)\n      _render_e3dp = _bare(Encoder_3D_Predictor)\n      _render_dec = _bare(Decoder_3D)\n\n      _render_efe.eval()\n      _render_e3dp.eval()\n      _render_dec.eval()\n"
        if old not in text:
            die("cannot find y=None / pred_images_3d=None in render block")
        text = text.replace(old, new, 1)
    # Use bare models for rendering.
    text = text.replace(
        "latent_code_3d=Encoder_3D_Predictor(Encoder_Feature_Extractor(y))[:,6:latent_code_dim_3d+6]",
        "latent_code_3d=_render_e3dp(_render_efe(y))[:,6:latent_code_dim_3d+6]",
    )
    text = text.replace(
        "test_pred_images_3D[slicei:slicei+1,...]=Decoder_3D(coords_3d[slicei:slicei+1,...]) if latent_code_3d==None else Decoder_3D(coords_3d[slicei:slicei+1,...],latent_code_3d)",
        "test_pred_images_3D[slicei:slicei+1,...]=_render_dec(coords_3d[slicei:slicei+1,...]) if latent_code_3d is None else _render_dec(coords_3d[slicei:slicei+1,...],latent_code_3d)",
    )
    # Add bottom barrier before leaving epoch loop.
    if "# DDP save/render sync: all ranks wait until writer finishes" not in text:
        old = "          test_pred_images_3D=None\n          torch.cuda.empty_cache()\n  end_time=time.time()"
        new = "          test_pred_images_3D=None\n          torch.cuda.empty_cache()\n\n    if (epoch+1)%args.save_interval==0:\n      # DDP save/render sync: all ranks wait until writer finishes.\n      distr.barrier(group=sync_pg)\n  end_time=time.time()"
        if old not in text:
            die("cannot find end of render block before end_time=time.time()")
        text = text.replace(old, new, 1)
    return text


def sanity_check(text: str):
    required = [
        "device = torch.device(f\"cuda:{local_rank}\")",
        "sync_pg = distr.new_group",
        "def get_global_min_free_mem_gb",
        "def ddp_wrap",
        "broadcast_buffers=False",
        "get_global_min_free_mem_gb(device)",
        "_render_dec",
        "distr.barrier(group=sync_pg)",
    ]
    missing = [s for s in required if s not in text]
    if missing:
        die("sanity check failed; missing: " + ", ".join(missing))


def main():
    if len(sys.argv) != 3:
        die("usage: python patch_cryofifa_ddpfix_v2.py cryofifa.py cryofifa_ddpfix.py")
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    if not inp.exists():
        die(f"input file not found: {inp}")
    text = inp.read_text()
    text = add_import_datetime(text)
    text = add_helpers(text)
    text = replace_setup(text)
    text = patch_ddp_calls(text)
    text = patch_memory_queries(text)
    text = patch_save_pose(text)
    text = patch_render_block(text)
    sanity_check(text)
    outp.write_text(text)
    print(f"Wrote {outp}")
    print("Next:")
    print(f"  python -m py_compile {outp}")
    print("  replace ../src/cryofifa.py with ../src/cryofifa_ddpfix.py in your run script")


if __name__ == "__main__":
    main()
