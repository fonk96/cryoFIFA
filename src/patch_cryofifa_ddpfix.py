#!/usr/bin/env python3
"""
Generate cryofifa_ddpfix.py from cryofifa.py with DDP/NCCL fixes.
Run this in the same directory as cryofifa.py:

    python patch_cryofifa_ddpfix.py cryofifa.py cryofifa_ddpfix.py

Then run cryofifa_ddpfix.py instead of cryofifa.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, name: str, required: bool = True) -> str:
    if old not in text:
        if required:
            raise RuntimeError(f"Cannot find block for: {name}")
        return text
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str, name: str, required: bool = True) -> str:
    count = text.count(old)
    if count == 0:
        if required:
            raise RuntimeError(f"Cannot find block for: {name}")
        return text
    return text.replace(old, new)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cryofifa.py")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cryofifa_ddpfix.py")

    text = src.read_text()

    # 1) import datetime
    if "import datetime" not in text:
        text = replace_once(text, "import time\n", "import time\nimport datetime\n", "add import datetime")

    # 2) helper functions before main
    helper = r'''

def get_global_min_free_mem_gb(device):
  """Query only this rank's GPU, then all-reduce the global minimum.
  Do not loop over torch.cuda.mem_get_info(gpui), otherwise every rank creates
  CUDA contexts on every GPU and can destabilize NCCL/DDP.
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
        text = replace_once(text, "\ndef main():\n", helper + "\ndef main():\n", "insert helper functions")

    # 3) distributed setup block
    old_setup = """  local_rank=int(os.environ[\"LOCAL_RANK\"])
  setup_for_distributed(local_rank==0)
  torch.cuda.set_device(local_rank)
  distr.init_process_group(\"nccl\",world_size=args.world_size,rank=local_rank)
  device=torch.device('cuda')
"""
    new_setup = """  local_rank = int(os.environ[\"LOCAL_RANK\"])
  rank = int(os.environ[\"RANK\"])
  world_size = int(os.environ[\"WORLD_SIZE\"])

  setup_for_distributed(rank == 0)

  torch.cuda.set_device(local_rank)
  device = torch.device(f\"cuda:{local_rank}\")

  distr.init_process_group(
    backend=\"nccl\",
    init_method=\"env://\",
  )

  # CPU-side sync group for long single-rank render/checkpoint sections.
  # This prevents other ranks from entering NCCL collectives while one rank writes maps.
  sync_pg = distr.new_group(
    backend=\"gloo\",
    timeout=datetime.timedelta(hours=12),
  )
"""
    text = replace_once(text, old_setup, new_setup, "distributed setup")

    # 4) DDP wrappers
    ddp_replacements = {
        "Encoder_Feature_Extractor=DDP(Encoder_Feature_Extractor,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Encoder_Feature_Extractor = ddp_wrap(Encoder_Feature_Extractor, local_rank, args)",
        "Encoder_Feature_Extractor_Local=DDP(Encoder_Feature_Extractor_Local,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Encoder_Feature_Extractor_Local = ddp_wrap(Encoder_Feature_Extractor_Local, local_rank, args)",
        "Encoder_Feature_Extractor_3d=DDP(Encoder_Feature_Extractor_3d,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Encoder_Feature_Extractor_3d = ddp_wrap(Encoder_Feature_Extractor_3d, local_rank, args)",
        "Encoder_3D_Predictor=DDP(Encoder_3D_Predictor,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Encoder_3D_Predictor = ddp_wrap(Encoder_3D_Predictor, local_rank, args)",
        "Encoder_3D_Predictor_Local=DDP(Encoder_3D_Predictor_Local,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Encoder_3D_Predictor_Local = ddp_wrap(Encoder_3D_Predictor_Local, local_rank, args)",
        "Decoder_3D=DDP(Decoder_3D,device_ids=[local_rank],gradient_as_bucket_view=args.fast_ddp)":
            "Decoder_3D = ddp_wrap(Decoder_3D, local_rank, args)",
    }
    for old, new in ddp_replacements.items():
        text = replace_all(text, old, new, f"DDP wrapper {old[:40]}")

    # 5) GPU memory query fixes
    old_mem_init = """  free_gpus_memorys=np.zeros(args.world_size)
  for gpui in range(args.world_size):
    free_gpus_memorys[gpui]=(torch.cuda.mem_get_info(gpui)[0]/1024/1024/1024)

  current_free_mem=min(free_gpus_memorys)
"""
    new_mem_init = """  current_free_mem = get_global_min_free_mem_gb(device)
"""
    text = replace_once(text, old_mem_init, new_mem_init, "initial free GPU memory query")

    old_mem_epoch = """    for gpui in range(args.world_size):
      free_gpus_memorys[gpui]=(torch.cuda.mem_get_info(gpui)[0]/1024/1024/1024)
      #current_free_mem=min(free_gpus_memorys)
    current_free_mem=min(free_gpus_memorys)
"""
    new_mem_epoch = """    current_free_mem = get_global_min_free_mem_gb(device)
"""
    text = replace_once(text, old_mem_epoch, new_mem_epoch, "epoch-end free GPU memory query")

    # 6) pose writing: avoid multi-rank append to same file
    old_pose_name = """            out_pose_name=args.render_path+'//'+args.output_pose_file+'_'+str(epoch+1)+'.star'
"""
    new_pose_name = """            out_pose_name=args.render_path+'//'+args.output_pose_file+'_'+str(epoch+1)+f'_rank{rank}.star'
"""
    text = replace_once(text, old_pose_name, new_pose_name, "rank-specific pose output name", required=False)

    # 7) save/render barrier and rank condition
    old_save_if = """    if (epoch+1)%args.save_interval==0 and local_rank==args.write_device :

      y=None
      pred_images_3d=None

      with torch.no_grad():
"""
    new_save_if = """    if (epoch+1)%args.save_interval==0:
      distr.barrier(group=sync_pg)

    if (epoch+1)%args.save_interval==0 and rank==args.write_device :

      y=None
      pred_images_3d=None

      _render_efe = _bare(Encoder_Feature_Extractor)
      _render_e3dp = _bare(Encoder_3D_Predictor)
      _render_dec = _bare(Decoder_3D)

      _render_efe.eval()
      _render_e3dp.eval()
      _render_dec.eval()

      with torch.inference_mode():
"""
    text = replace_once(text, old_save_if, new_save_if, "save/render block entry")

    # Render should use bare modules, not DDP wrappers, on the single writer rank.
    old_latent = """            latent_code_3d=Encoder_3D_Predictor(Encoder_Feature_Extractor(y))[:,6:latent_code_dim_3d+6]
"""
    new_latent = """            latent_code_3d=_render_e3dp(_render_efe(y))[:,6:latent_code_dim_3d+6]
"""
    text = replace_once(text, old_latent, new_latent, "render latent with bare modules", required=False)

    old_dec = """            test_pred_images_3D[slicei:slicei+1,...]=Decoder_3D(coords_3d[slicei:slicei+1,...]) if latent_code_3d==None else Decoder_3D(coords_3d[slicei:slicei+1,...],latent_code_3d)
"""
    new_dec = """            test_pred_images_3D[slicei:slicei+1,...]=_render_dec(coords_3d[slicei:slicei+1,...]) if latent_code_3d is None else _render_dec(coords_3d[slicei:slicei+1,...],latent_code_3d)
"""
    text = replace_once(text, old_dec, new_dec, "render decoder with bare module", required=False)

    # Add post-render barrier after the exact final cleanup line in the save block.
    old_cleanup = """          test_pred_images_3D=None
          torch.cuda.empty_cache()
  end_time=time.time()
"""
    new_cleanup = """          test_pred_images_3D=None
          torch.cuda.empty_cache()

    if (epoch+1)%args.save_interval==0:
      distr.barrier(group=sync_pg)
  end_time=time.time()
"""
    text = replace_once(text, old_cleanup, new_cleanup, "save/render block exit barrier")

    # 8) A couple of conservative identity fixes.
    text = text.replace("if latent_code_3d==None:", "if latent_code_3d is None:")
    text = text.replace("if latent_code_3d!=0:", "if latent_code_dim_3d!=0:") if False else text

    dst.write_text(text)
    print(f"Wrote {dst}")
    print("Run a syntax check with:")
    print(f"  python -m py_compile {dst}")
    print("Then launch torchrun using this new file instead of cryofifa.py.")


if __name__ == "__main__":
    main()
