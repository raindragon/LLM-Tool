"""FTW 权重数值对拍（通用版）：HF safetensors  vs  FTW，覆盖 NVFP4 与 FP8 两条路。

用法:
    python verify_ftw_weights.py nvfp4 0 mlp.down_proj
    python verify_ftw_weights.py fp8   0 linear_attn.out_proj
    python verify_ftw_weights.py fp8fuse 0 linear_attn.in_proj_qkvz
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open

HF_DIR = r"G:\AI\Qwen3.8-27B-NVFP4"
FTW_DIR = r"G:\AI\Qwen3.8-27B-NVFP4-FTW"
DTYPE_MAP = {"float8_e4m3fn": torch.float8_e4m3fn, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16, "uint8": torch.uint8}
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4T = torch.cat([E2M1, -E2M1])

MANIFEST = json.load(open(os.path.join(FTW_DIR, "freetoken_weight.json"), encoding="utf-8"))
FT = {t["name"]: t for t in MANIFEST["tensors"]}
SHARDS = MANIFEST["shards"]
WM = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]


def ftw(name: str):
    if name not in FT:
        return None
    e = FT[name]
    off, nb = e["global_off"], e["nbytes"]
    for sh in SHARDS:
        if sh["global_off"] <= off < sh["global_off"] + sh["nbytes"]:
            with open(os.path.join(FTW_DIR, sh["file"]), "rb") as f:
                f.seek(off - sh["global_off"])
                buf = f.read(nb)
            raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
            return raw.view(DTYPE_MAP[e["dtype"]]).reshape(e["shape"])
    return None


def hf(name: str):
    fn = WM.get(name)
    if fn is None:
        return None
    with safe_open(os.path.join(HF_DIR, fn), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def deq_nvfp4(w, s, g, n, k):
    lo = (w & 0x0F).to(torch.long)
    hi = ((w >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([FP4T[lo], FP4T[hi]], dim=-1).reshape(n, k)
    s = s.to(torch.float32)
    if s.dim() == 1:
        s = s[:, None]
    return codes * s.repeat_interleave(16, 1) * torch.as_tensor(g, dtype=torch.float32).reshape(-1, 1)


def stats(t, tag):
    t = t.to(torch.float32)
    print(f"   {tag:34s} {tuple(t.shape)!s:18s} mean={t.mean():+.6g} absmax={t.abs().max():.6g}")


def check_nvfp4(layer: int, path: str) -> None:
    hb = f"model.language_model.layers.{layer}.{path}"
    fb = f"model.layers.{layer}.{path}"
    hw, hs, hg = hf(hb + ".weight"), hf(hb + ".weight_scale"), hf(hb + ".weight_scale_2")
    fw, fs, fg = ftw(fb + ".weight"), ftw(fb + ".weight_scale"), ftw(fb + ".weight_global")
    print(f"[NVFP4] {path}  (layer {layer})")
    for tag, t in (("HF weight", hw), ("FTW weight", fw), ("HF w_scale", hs),
                   ("FTW w_scale", fs), ("HF w_scale_2", hg), ("FTW weight_global", fg)):
        if t is not None:
            stats(t, tag)
    if hw is None or fw is None:
        print("   !! 缺少张量"); return
    n, k = hw.shape[0], hw.shape[1] * 2
    ref = deq_nvfp4(hw, hs, hg, n, k)
    got = deq_nvfp4(fw, fs, fg, n, k)
    d = (ref - got).abs()
    print(f"   -> 反量化 max|Δ|={float(d.max()):.6g}  相对={float(d.max()/ref.abs().max()):.4g}"
          f"   (weight字节同={torch.equal(hw.flatten(), fw.flatten())})")


def check_fp8(layer: int, path: str) -> None:
    hb = f"model.language_model.layers.{layer}.{path}"
    fb = f"model.layers.{layer}.{path}"
    hw, hs, hi = hf(hb + ".weight"), hf(hb + ".weight_scale"), hf(hb + ".input_scale")
    fw, fs, fi = ftw(fb + ".weight"), ftw(fb + ".weight_scale"), ftw(fb + ".input_scale")
    print(f"[FP8 ] {path}  (layer {layer})")
    for tag, t in (("HF weight", hw), ("FTW weight", fw), ("HF w_scale", hs),
                   ("FTW w_scale", fs), ("HF input_scale", hi), ("FTW input_scale", fi)):
        if t is not None:
            stats(t, tag)
    if hw is None or fw is None:
        print("   !! 缺少张量"); return
    ok_w = torch.equal(hw.flatten(), fw.flatten())
    n = hw.shape[0]
    hs_v, fs_v = float(torch.as_tensor(hs).flatten()[0]), float(fs.flatten()[0])
    print(f"   -> weight字节同={ok_w}  HF标量={hs_v:.8g}  FTW[0]={fs_v:.8g} "
          f"FTW是否全等={bool((fs == fs.flatten()[0]).all())}  n={n}")
    if hi is not None and fi is not None:
        print(f"   -> input_scale HF={float(torch.as_tensor(hi).flatten()[0]):.8g} "
              f"FTW={float(fi.flatten()[0]):.8g}")


def check_fp8_fuse(layer: int, fused_path: str, parts: list[str]) -> None:
    fb = f"model.layers.{layer}.{fused_path}"
    fw, fs = ftw(fb + ".weight"), ftw(fb + ".weight_scale")
    hb = f"model.language_model.layers.{layer}."
    print(f"[FP8f] {fused_path}  <= {parts}  (layer {layer})")
    rows, scales = [], []
    for p in parts:
        pw, ps = hf(hb + p + ".weight"), hf(hb + p + ".weight_scale")
        if pw is None:
            print(f"   !! HF 缺 {p}.weight"); return
        rows.append(pw)
        scales.append(float(torch.as_tensor(ps).flatten()[0]))
        print(f"   part {p:58s} {tuple(pw.shape)!s:14s} scale={float(torch.as_tensor(ps).flatten()[0]):.6g}")
    ref_w = torch.cat([r.view(torch.uint8) for r in rows], 0).view(torch.float8_e4m3fn)
    exp_scale = torch.cat([torch.full((r.shape[0],), s, dtype=torch.float32)
                           for r, s in zip(rows, scales)])
    print(f"   融合后: HF拼接={tuple(ref_w.shape)}  FTW={tuple(fw.shape)}  "
          f"weight字节同={torch.equal(ref_w.flatten(), fw.flatten())}")
    print(f"   scale 行分段是否一致={torch.equal(exp_scale, fs.flatten())}")
    if not torch.equal(exp_scale, fs.flatten()):
        bad = (exp_scale != fs.flatten()).nonzero().flatten()
        print(f"      不一致行数={bad.numel()} 前几处 idx={bad[:6].tolist()} "
              f"期望={exp_scale[bad[:6]].tolist()} 实际={fs.flatten()[bad[:6]].tolist()}")
    stats(fw, "FTW weight")


def main() -> int:
    kind = sys.argv[1]
    layer = int(sys.argv[2])
    tail = sys.argv[3]
    if kind == "nvfp4":
        check_nvfp4(layer, tail)
    elif kind == "fp8":
        check_fp8(layer, tail)
    elif kind == "fp8fuse":
        parts = sys.argv[4:]
        check_fp8_fuse(layer, tail, parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
