#!/usr/bin/env python3
# Robust text patcher for cryofifa.py DDP/NCCL issues.
# Usage: python patch_cryofifa_ddpfix_v3.py cryofifa.py cryofifa_ddpfix.py

from pathlib import Path
import re
import sys


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def replace_regex(text: str, pattern: str, repl: str, name: str, count: int = 1, flags: int = 0) -> str:
    new, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n == 0:
        die(f"cannot patch block: {name}")
    print(f"patched {name}: {n}")
    return new


def main() -> None:
    if len(sys.argv) != 3:
        die("usage: python patch_cryofifa_ddpfix_v3.py cryofifa.py cryofifa_ddpfix.py")

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        die(f"input file not found: {src}")

    text = src.read_text()

    # 1) import datetime for long-timeout Gloo sync group.
    if "import datetime" not in text:
        if re.search(r"^import time\s*$", text, flags=re.M):
            text = re.sub(r"^(import time\s*)$", r"\1\nimport datetime", text, count=1, flags=re.M)
            print("patched import datetime")
        else:
            # Safe fallback: add near top after future import block.
            text = "import datetime\n" + text
            print("patched import datetime at file top")

    # 2) add helper functions after load_checkpoint return.
    helper = r'''


def get_global_min_free_mem_gb(device):
  """Query only the current rank's GPU, then all-reduce the global minimum.
  This avoids creating CUDA contexts on every GPU from every rank.
  """
  free_bytes, _ = torch.cuda.mem_get_info(device)
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
    if "def get_global_min_free_mem_gb" not in text:
        pat = r"(\n\s*return\s+int\(ckpt\.get\('epoch',\s*-1\)\)\s*\+\s*1\s*#\s*resume at the next epoch\s*\n)"
        text = replace_regex(text, pat, r"\1" + helper + "\n", "helper functions", flags=re.M)

    # 3) distributed setup: replace between setup_distributed banner and next ===== banner.
    setup_repl = '''  #===========setup_distributed==========#

  local_rank = int(os.environ["LOCAL_RANK"])
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  setup_for_distributed(rank == 0)

  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")

  distr.init_process_group(
    backend="nccl",
    init_method="env://",
  )

  # CPU-side synchronization group for long single-rank render/checkpoint sections.
  # This prevents other ranks from entering NCCL collectives while one rank is still writing maps.
  sync_pg = distr.new_group(
    backend="gloo",
    timeout=datetime.timedelta(hours=12),
  )

  #======================================#'''
    text = replace_regex(
        text,
        r"\s*#=+setup_distributed=+#\s*\n.*?\n\s*#=+=+#",
        "\n" + setup_repl,
        "distributed setup",
        flags=re.S,
    )

    # 4) replace one-line DDP wrappers with ddp_wrap.
    ddp_names = [
        "Encoder_Feature_Extractor",
        "Encoder_Feature_Extractor_Local",
        "Encoder_Feature_Extractor_3d",
        "Encoder_3D_Predictor",
        "Encoder_3D_Predictor_Local",
        "Decoder_3D",
    ]
    for name in ddp_names:
        # Allows arbitrary keyword args in the original one-line DDP call.
        pat = rf"^(\s*){name}\s*=\s*DDP\(\s*{name}\s*,\s*device_ids\s*=\s*\[local_rank\].*?\)\s*$"
        repl = rf"\1{name} = ddp_wrap({name}, local_rank, args)"
        text, n = re.subn(pat, repl, text, flags=re.M)
        if n:
            print(f"patched DDP wrapper {name}: {n}")

    # 5) replace memory query blocks. These regexes are tolerant of spaces/comments.
    text, n1 = re.subn(
        r"\n\s*free_gpus_memorys\s*=\s*np\.zeros\(args\.world_size\)\s*\n\s*for\s+gpui\s+in\s+range\(args\.world_size\)\s*:\s*\n\s*free_gpus_memorys\[gpui\]\s*=\s*\(\s*torch\.cuda\.mem_get_info\(gpui\)\[0\]\s*/\s*1024\s*/\s*1024\s*/\s*1024\s*\)\s*\n\s*current_free_mem\s*=\s*min\(free_gpus_memorys\)\s*\n",
        "\n  current_free_mem = get_global_min_free_mem_gb(device)\n",
        text,
        count=1,
        flags=re.S,
    )
    print(f"patched initial memory query block: {n1}")

    text, n2 = re.subn(
        r"\n\s*for\s+gpui\s+in\s+range\(args\.world_size\)\s*:\s*\n\s*free_gpus_memorys\[gpui\]\s*=\s*\(\s*torch\.cuda\.mem_get_info\(gpui\)\[0\]\s*/\s*1024\s*/\s*1024\s*/\s*1024\s*\)\s*\n\s*(?:#.*\n)?\s*current_free_mem\s*=\s*min\(free_gpus_memorys\)\s*\n",
        "\n    current_free_mem = get_global_min_free_mem_gb(device)\n",
        text,
        count=1,
        flags=re.S,
    )
    print(f"patched epoch-end memory query block: {n2}")

    # Fallback: if a mem_get_info(gpui) occurrence remains, make it query this rank's device.
    # This is safe and prevents cross-GPU context creation even if a surrounding loop remains.
    text, n3 = re.subn(r"torch\.cuda\.mem_get_info\(\s*gpui\s*\)", "torch.cuda.mem_get_info(device)", text)
    if n3:
        print(f"fallback-patched remaining mem_get_info(gpui): {n3}")

    # 6) save_pose: avoid all ranks appending the same STAR file.
    text, n = re.subn(
        r"out_pose_name\s*=\s*args\.render_path\s*\+\s*'//'\s*\+\s*args\.output_pose_file\s*\+\s*'_'\s*\+\s*str\(epoch\+1\)\s*\+\s*'\.star'",
        "out_pose_name=args.render_path+'//'+args.output_pose_file+'_'+str(epoch+1)+f'_rank{rank}.star'",
        text,
    )
    if n:
        print(f"patched save_pose output name: {n}")

    # 7) render/checkpoint block: add barrier and use global rank, not local_rank.
    text, n = re.subn(
        r"^(\s*)if\s*\(epoch\+1\)%args\.save_interval==0\s+and\s+local_rank==args\.write_device\s*:",
        r"\1if (epoch+1)%args.save_interval==0:\n\1  distr.barrier(group=sync_pg)\n\n\1if (epoch+1)%args.save_interval==0 and rank==args.write_device :",
        text,
        count=1,
        flags=re.M,
    )
    if n == 0:
        die("cannot patch render/save if block")
    print("patched render/save entry barrier")

    # Add bare-model aliases inside render block after y/pred_images initialization.
    text, n = re.subn(
        r"(\n\s*y\s*=\s*None\s*\n\s*pred_images_3d\s*=\s*None\s*\n)",
        r"\1\n      _render_efe = _bare(Encoder_Feature_Extractor)\n      _render_e3dp = _bare(Encoder_3D_Predictor)\n      _render_dec = _bare(Decoder_3D)\n\n      _render_efe.eval()\n      _render_e3dp.eval()\n      _render_dec.eval()\n",
        text,
        count=1,
    )
    if n == 0:
        die("cannot insert bare render models")
    print("patched bare render models")

    # Replace DDP-forward calls inside render with bare module calls.
    text, n1 = re.subn(
        r"latent_code_3d\s*=\s*Encoder_3D_Predictor\(Encoder_Feature_Extractor\(y\)\)\s*\[:,6:latent_code_dim_3d\+6\]",
        "latent_code_3d=_render_e3dp(_render_efe(y))[:,6:latent_code_dim_3d+6]",
        text,
    )
    print(f"patched render latent forward: {n1}")

    text, n2 = re.subn(
        r"test_pred_images_3D\[slicei:slicei\+1,\.\.\.\]\s*=\s*Decoder_3D\(coords_3d\[slicei:slicei\+1,\.\.\.\]\)\s*if\s*latent_code_3d==None\s*else\s*Decoder_3D\(coords_3d\[slicei:slicei\+1,\.\.\.\],latent_code_3d\)",
        "test_pred_images_3D[slicei:slicei+1,...]=_render_dec(coords_3d[slicei:slicei+1,...]) if latent_code_3d is None else _render_dec(coords_3d[slicei:slicei+1,...],latent_code_3d)",
        text,
    )
    print(f"patched render decoder forward: {n2}")

    # 8) add exit barrier after render block. Patch the last occurrence before end_time.
    pat = r"(\n\s*test_pred_images_3D\s*=\s*None\s*\n\s*torch\.cuda\.empty_cache\(\)\s*\n)(\s*end_time\s*=\s*time\.time\(\))"
    repl = r"\1\n    if (epoch+1)%args.save_interval==0:\n      distr.barrier(group=sync_pg)\n\2"
    text, n = re.subn(pat, repl, text, count=1)
    if n == 0:
        die("cannot patch render/save exit barrier")
    print("patched render/save exit barrier")

    # 9) Ensure common unsafe global device shorthand is reduced in this file only where easy.
    # We do NOT blindly replace all .cuda() calls because user code may intentionally manage some objects.
    # But device is now cuda:<local_rank>, so .to(device) is safe.

    # Final sanity diagnostics; do not abort for minor leftovers except the worst cross-GPU query.
    if "mem_get_info(gpui)" in text:
        die("still found torch.cuda.mem_get_info(gpui); patch incomplete")
    if "distr.init_process_group(\"nccl\",world_size=args.world_size,rank=local_rank)" in text:
        die("old init_process_group still present")

    dst.write_text(text)
    print(f"WROTE: {dst}")
    print("Next: python -m py_compile", dst)


if __name__ == "__main__":
    main()
