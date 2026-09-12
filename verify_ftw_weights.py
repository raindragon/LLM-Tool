"""对拍脚本：同一权重分别从 HF safetensors 与 FTW 反量化，逐元素比较。

只测非融合的 NVFP4 叶子层 (mlp.down_proj)，路径最短，最容易定位。

反量化公式 (ModelOpt NVFP4):
    W[n,k] = fp4_e2m1(code) * weight_scale[n, k//16] * weight_scale_2
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open

HF_DIR = r"G:\AI\Qwen3.8-27B-NVFP4"
FTW_DIR = r"G:\AI\Qwen3.8-27B-NVFP4-FTW"

# e2m1 码表: 低3位 = 幅值码, 第4位 = 符号
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def fp4_table() -> torch.Tensor:
    return torch.cat([E2M1, -E2M1])  # index 0..15


DTYPE_MAP = {"float8_e4m3fn": torch.float8_e4m3fn, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16, "uint8": torch.uint8}


def decode_nvfp4(packed: torch.Tensor, scale: torch.Tensor, gscale, n: int, k: int,
                 high_first: bool = False) -> torch.Tensor:
    """packed uint8 [n, k//2] -> fp32 [n, k]"""
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    tbl = fp4_table()
    a = tbl[hi] if high_first else tbl[lo]
    b = tbl[lo] if high_first else tbl[hi]
    codes = torch.stack([a, b], dim=-1).reshape(n, k)          # 偶数位, 奇数位
    s = scale.to(torch.float32)
    if s.dim() == 1:                                            # [n] 每行一个 scale
        s = s[:, None]
    assert s.shape == (n, k // 16), f"scale shape {tuple(s.shape)} != ({n},{k//16})"
    s = s.repeat_interleave(16, dim=1)
    g = torch.as_tensor(gscale, dtype=torch.float32)
    if g.dim() == 1:
        g = g[:, None]
    return codes * s * g


def ftw_read(name: str, manifest: dict, shards: list[dict]) -> torch.Tensor:
    e = next(t for t in manifest["tensors"] if t["name"] == name)
    off, nb = e["global_off"], e["nbytes"]
    for sh in shards:
        if sh["global_off"] <= off < sh["global_off"] + sh["nbytes"]:
            local = off - sh["global_off"]
            with open(os.path.join(FTW_DIR, sh["file"]), "rb") as f:
                f.seek(local)
                buf = f.read(nb)
            raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
            return raw.view(DTYPE_MAP[e["dtype"]]).reshape(e["shape"]), e
    raise KeyError(f"{name} 不在任何分片中")


def as_dtype(raw: torch.Tensor, dt: str) -> torch.Tensor:
    m = {"float8_e4m3fn": torch.float8_e4m3fn, "float32": torch.float32,
         "float16": torch.float16, "bfloat16": torch.bfloat16, "uint8": torch.uint8}
    return raw.view(m[dt])


def main() -> int:
    layer = sys.argv[1] if len(sys.argv) > 1 else "0"
    base = f"model.language_model.layers.{layer}.mlp.down_proj"
    manifest = json.load(open(os.path.join(FTW_DIR, "freetoken_weight.json"), encoding="utf-8"))
    shards = manifest["shards"]

    # ---- HF 侧 ----
    index = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json"), encoding="utf-8"))
    wm = index["weight_map"]
    hf = {}
    for suf in (".weight", ".weight_scale", ".weight_scale_2"):
        key = base + suf
        fn = wm.get(key)
        if fn is None:
            print(f"! HF 缺 {key}")
            return 1
        with safe_open(os.path.join(HF_DIR, fn), framework="pt", device="cpu") as f:
            hf[suf] = f.get_tensor(key)
    print("HF 原始:")
    for k, v in hf.items():
        print(f"   {k:16s} {str(v.dtype):16s} {tuple(v.shape)}  "
              f"min={float(v.to(torch.float32).min()):.6g} max={float(v.to(torch.float32).max()):.6g}")

    print("\nFTW 侧:")
    ftw = {}
    for suf in (".weight", ".weight_scale", ".weight_global"):
        raw, e = ftw_read(f"model.layers.{layer}.mlp.down_proj{suf}", manifest, shards)
        ftw[suf] = as_dtype(raw, e["dtype"])
        print(f"   {suf:16s} {e['dtype']:16s} {tuple(e['shape'])}  "
              f"min={float(ftw[suf].to(torch.float32).min()):.6g} max={float(ftw[suf].to(torch.float32).max()):.6g}")

    n, k = hf[".weight"].shape[0], hf[".weight"].shape[1] * 2
    print(f"\n逻辑尺寸 n={n} k={k}")

    # ---- 反量化对拍 ----
    ref = decode_nvfp4(hf[".weight"], hf[".weight_scale"], hf[".weight_scale_2"], n, k)
    for hf_order in (False, True):
        got = decode_nvfp4(ftw[".weight"], ftw[".weight_scale"], ftw[".weight_global"], n, k,
                           high_first=hf_order)
        diff = (ref - got).abs()
        rel = diff.max() / ref.abs().max()
        tag = "高半字节在前" if hf_order else "低半字节在前"
        print(f"   [{tag}] max|diff|={float(diff.max()):.6g}  mean|diff|={float(diff.mean()):.6g}  "
              f"相对={float(rel):.4g}  ref范围={float(ref.abs().max()):.4g}")

    # ---- 独立校验: 两个 scale 集合是否一致 ----
    s_ref = hf[".weight_scale"].to(torch.float32).flatten()
    s_ftw = ftw[".weight_scale"].to(torch.float32).flatten()
    print(f"\nweight_scale 完全相同: {torch.equal(s_ref, s_ftw)}"
          f"  max|diff|={(s_ref - s_ftw).abs().max():.6g}")
    g_ref = float(torch.as_tensor(hf[".weight_scale_2"]).flatten()[0])
    g_ftw = float(ftw[".weight_global"].flatten()[0])
    print(f"global scale: HF={g_ref:.8g}  FTW={g_ftw:.8g}  一致={abs(g_ref - g_ftw) < 1e-9}")
    print(f"weight 字节完全相同: {torch.equal(hf['.weight'].flatten(), ftw['.weight'].flatten())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
