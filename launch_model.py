#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""FreeToken 本地模型一键启动

绕过 FreeToken Desktop，直接用 ft.exe 起一个 OpenAI 兼容服务。

    python launch_model.py                          # 默认：自动找模型/自动算显存
    python launch_model.py --model G:\AI\xxx-FTW
    python launch_model.py --mode low-mem           # 省显存模式
    python launch_model.py --mode long-ctx          # 长上下文模式
    python launch_model.py --port 8088 --concurrency 8
    python launch_model.py --dry-run                # 只打印命令不启动

设计要点：
  * memory-ratio 是这台的唯一坑：预算是
        KV = ratio x 加载前空闲显存 - 权重 - 固定缓存
    权重 ~18.8GiB 时 ratio 给 0.85 会算成负数，直接
    "Not enough memory for KV cache"。本脚本按实际空闲显存反推安全值。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1024 ** 3

MODES = {
    # 名称:        (并发, prefill, 缓存策略覆盖)
    "default":     (4, 4096, None),
    "low-mem":     (1, 2048, "naive"),
    "long-ctx":    (2, 4096, None),
    "high-conc":   (8, 2048, "naive"),
}


# --------------------------------------------------------------------------- #
# 显存预算（混合线性注意力模型的关键）
# --------------------------------------------------------------------------- #

def _text_config(model: str) -> dict:
    cfg = os.path.join(model, "config.json")
    if not os.path.isfile(cfg):
        return {}
    try:
        with open(cfg, "r", encoding="utf-8") as fh:
            j = json.load(fh)
    except Exception:
        return {}
    return j.get("text_config") or j


def gdn_state_per_req(text_cfg: dict, dtype_bytes: int = 2) -> tuple[int, int]:
    """单个请求的 GDN（线性注意力）状态字节数 + 线性层数。"""
    layers = text_cfg.get("layer_types") or []
    n_linear = sum(1 for t in layers if t == "linear_attention")
    if not n_linear:
        return 0, 0
    nk = text_cfg.get("linear_num_key_heads", 0)
    kd = text_cfg.get("linear_key_head_dim", 0)
    nv = text_cfg.get("linear_num_value_heads", 0)
    vd = text_cfg.get("linear_value_head_dim", 0)
    ker = text_cfg.get("linear_conv_kernel_dim", 4)
    conv_bytes = (nk * kd + nv * vd) * max(0, ker - 1) * dtype_bytes
    rec_bytes = nv * kd * vd * 4
    return n_linear * (conv_bytes + rec_bytes), n_linear


def state_slots(conc: int, cache_type: str, cache_ratio: float = 2.0) -> int:
    """引擎 _linear_pool_num_slots：GDN 状态池的物理槽位数。

    radix 下是 ``4 x 并发 + max(4, 2 x 并发) + 1`` —— 并发 1 只要 9 槽，
    并发 4 直接跳到 25 槽。这是这种模型上最容易踩的显存坑。
    """
    mr = max(1, int(conc))
    if cache_type != "radix":
        return mr + 1
    return 4 * mr + max(4, int(cache_ratio * mr)) + 1


def plan_launch(weights: int, free: int, ratio: float, want_conc: int,
                want_prefill: int, cache_pref: str | None,
                per_req: int, kv_floor: int = 1024 ** 3) -> dict:
    """在预算内选出可用的 (并发, 缓存, prefill)，并给出说明。

    预算：pool = ratio x free - weights；KV = pool - GDN 状态池。
    KV 一旦 <= 0，引擎就抛 "Not enough memory for KV cache"。
    """
    pool = int(ratio * free) - weights if free else 0
    candidates = []
    for conc in sorted({want_conc, 8, 4, 2, 1}, reverse=True):
        for ct in ("radix", "naive"):
            if cache_pref and ct != cache_pref and conc == want_conc:
                # 用户/模式显式指定了缓存策略，先按这个试
                continue
            slots = state_slots(conc, ct)
            kv = pool - per_req * slots
            candidates.append({
                "conc": conc, "cache": ct, "slots": slots,
                "state": per_req * slots, "kv": kv, "ok": bool(free) and kv >= kv_floor,
            })
    if cache_pref:
        candidates.sort(key=lambda c: (not c["ok"], c["cache"] != cache_pref,
                                       abs(c["conc"] - want_conc)))
    else:
        candidates.sort(key=lambda c: (not c["ok"], c["conc"] != want_conc,
                                       c["cache"] != "radix", -c["conc"]))
    usable = [c for c in candidates if c["ok"]]
    pick = usable[0] if usable else None
    if pick:
        pick = dict(pick)
        pick["prefill"] = want_prefill if free - weights > 3 * 1024 ** 3 else min(want_prefill, 2048)
        pick["all"] = candidates
        pick["pool"] = pool
    return pick or {"ok": False, "pool": pool, "all": candidates}


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #

def find_ft() -> str | None:
    cands = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        cands.append(os.path.join(local, "FreeToken", "venv", "Scripts", "ft.exe"))
    cands.append(r"C:\Program Files\FreeToken\venv\Scripts\ft.exe")
    cands.append(os.path.join(HERE, "ft.exe"))
    for p in cands:
        if os.path.isfile(p):
            return p
    return shutil.which("ft")


def find_model() -> str | None:
    """按优先级找可用的 FTW 模型目录。"""
    # 1) 环境变量
    env = os.environ.get("FTW_MODEL")
    if env and os.path.isdir(env):
        return env
    # 2) 脚本同级 / 上级的 *-FTW
    for base in (HERE, os.path.dirname(HERE), r"G:\AI", r"N:\AI", os.path.expanduser("~")):
        if not os.path.isdir(base):
            continue
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for n in names:
            p = os.path.join(base, n)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "freetoken_weight.json")):
                return p
    return None


def weights_bytes(model: str) -> int:
    total = 0
    for f in os.listdir(model):
        if f.endswith((".ftw", ".safetensors", ".gguf")):
            try:
                total += os.path.getsize(os.path.join(model, f))
            except OSError:
                pass
    return total


def gpu_free_bytes(index: int = 0) -> int | None:
    smi = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not smi and os.path.isfile(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        smi = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    try:
        cmd = [smi] if smi else ["nvidia-smi"]
        out = subprocess.run(
            cmd + ["--query-gpu=index,memory.total,memory.free",
                  "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0] == str(index):
            return int(float(parts[2])) * 1024 * 1024
    return None


def ratio_for(free: int, reserve: int, cap: float) -> float:
    """给定"预留余量"和上限，算出这一档的 memory-ratio。

    预算是 ``KV = ratio x free - weights - GDN状态池``；ratio 越高给 KV 的余量越大，
    但 ``(1 - ratio)`` 那一块要留给 CUDA graph 捕获 + 激活值。
    """
    if not free:
        return 0.90
    return round(min(cap, max(0.72, (free - reserve) / free)), 3)


# 两档显存预留：安全档优先；安全档全不满足时退极限档。
# safe       —— 留 3.5 GiB 或 15%，ratio ≤ 0.90（最稳）
# aggressive —— 留 1.5 GiB 或 6%，ratio ≤ 0.95，并关小 CUDA graph batch
# 不留极限档的话，「18.8G 权重 / 22.7G 空闲」这种差一点点的组合会被直接判死。
def ratio_tiers(free: int) -> list[tuple[str, int, float]]:
    if not free:
        return [("safe", 0, 0.90)]
    return [
        ("safe", int(max(3.5 * GIB, 0.15 * free)), 0.90),
        ("aggressive", int(max(1.5 * GIB, 0.06 * free)), 0.95),
    ]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="FreeToken 本地模型一键启动", add_help=True)
    ap.add_argument("--model", default=None, help="FTW（或 HF）模型目录")
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--gpu", default="0", help="GPU 索引（默认 0）")
    ap.add_argument("--mode", default="default", choices=sorted(MODES), help="预设模式")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--prefill", type=int, default=None)
    ap.add_argument("--ratio", default="auto", help="memory-ratio，auto=按显存预算反推")
    ap.add_argument("--cache", default=None, choices=[None, "radix", "naive"],
                    help="强制 KV 缓存策略（radix 更快，naive 更省 GDN 状态池）")
    ap.add_argument("--seq-len", type=int, default=None, help="最大序列长度覆盖")
    ap.add_argument("--extra", default="", help="追加给 ft serve 的参数")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    ap.add_argument("--no-wait", action="store_true", help="启动后不再等就绪")
    ap.add_argument("--log", default=None, help="日志文件路径")
    args = ap.parse_args()

    ft = find_ft()
    if not ft:
        print("[x] 找不到 ft.exe。请设置环境变量 FT_EXE 或安装 FreeToken Desktop。",
              file=sys.stderr)
        return 1

    model = os.path.abspath(args.model) if args.model else find_model()
    if not model or not os.path.isdir(model):
        print("[x] 找不到模型目录，请用 --model 指定。", file=sys.stderr)
        return 1

    conc_want, prefill_want, cache_pref = MODES[args.mode]
    if args.concurrency:
        conc_want = args.concurrency
    if args.prefill:
        prefill_want = args.prefill
    if args.cache:
        cache_pref = args.cache

    weights = weights_bytes(model)
    free = gpu_free_bytes(int(args.gpu))
    text_cfg = _text_config(model)
    per_req, n_linear = gdn_state_per_req(text_cfg)

    tier, note = "safe", ""
    plan = None
    if args.ratio != "auto":
        ratio = float(args.ratio)
        tier = "manual"
        note = f"手动指定 ratio {ratio}"
        plan = plan_launch(weights, free, ratio, conc_want, prefill_want, cache_pref, per_req)
    elif not free:
        ratio, note = 0.90, "读不到显存信息，回退 0.90"
        plan = plan_launch(weights, 0, ratio, conc_want, prefill_want, cache_pref, per_req)
    else:
        # 逐档试：真正能不能跑由 plan_launch 说了算（它算 GDN 状态池 + KV）
        for tname, reserve, cap in ratio_tiers(free):
            r = ratio_for(free, reserve, cap)
            p = plan_launch(weights, free, r, conc_want, prefill_want, cache_pref, per_req)
            ratio, tier, plan = r, tname, p
            note = (f"权重 {weights / GIB:.1f} GiB · 加载前空闲 {free / GIB:.1f} GiB "
                    f"→ ratio {r:.3f}（{tname} 档，图形/激活预留 "
                    f"{free * (1 - r) / GIB:.2f} GiB）")
            if p.get("ok"):
                break

    name = os.path.basename(model.rstrip("\\/"))
    print("=" * 70)
    print("  FreeToken 本地模型启动")
    print("=" * 70)
    print(f"  引擎    : {ft}")
    print(f"  模型    : {model}")
    print(f"  权重    : {weights / GIB:.2f} GiB")
    print(f"  显存    : {note}")

    if not plan.get("ok"):
        print("-" * 70)
        print("  ✗ 当前空闲显存装不下这个模型，试过的方案都不行：")
        print(f"    {'并发':>4} {'缓存':>6} {'GDN槽位':>8} {'状态池':>9} {'KV余量':>10}")
        for c in plan.get("all", []):
            print(f"    {c['conc']:>4} {c['cache']:>6} {c['slots']:>8} "
                  f"{c['state'] / GIB:>7.2f}G {c['kv'] / GIB:>9.2f}G")
        print("=" * 70)
        print("  处理办法：关掉其它占显存的程序 / 换更小的量化版本 / 用 --ratio 提高上限")
        return 1

    conc, cache, prefill = plan["conc"], plan["cache"], plan["prefill"]
    if n_linear:
        print(f"  线性层  : {n_linear} 层 GDN，单请求状态 {per_req / GIB:.2f} GiB")
    print(f"  模式    : {args.mode}  并发 {conc}  prefill {prefill}  缓存 {cache}")
    print(f"  预算    : 可用池 {plan['pool'] / GIB:.2f} GiB → "
          f"状态池 {plan['state'] / GIB:.2f} GiB + KV {plan['kv'] / GIB:.2f} GiB"
          f"（{plan['slots']} 槽位）")
    print(f"  接口    : http://127.0.0.1:{args.port}/v1")

    notes = []
    if tier == "aggressive":
        notes.append(f"⚠ 安全余量装不下，已切到极限档（ratio {ratio:.3f}）；"
                     f"CUDA graph batch 压到 1，长上下文/高并发仍可能 OOM")
    if conc != conc_want:
        notes.append(f"目标并发 {conc_want} 的 GDN 状态池放不下，已降到 {conc}")
    if cache_pref and cache != cache_pref:
        notes.append(f"缓存策略由 {cache_pref} 调整为 {cache}")
    if not cache_pref and conc_want >= 4 and cache != "radix":
        notes.append(f"并发 {conc} 用 radix 需要 {state_slots(conc, 'radix')} 个 GDN 槽位，"
                     f"改用 naive 才能塞进显存（代价：失去跨请求前缀复用）")
    for n in notes:
        print(f"  提示    : {n}")

    cmd = [
        ft, "serve",
        "--model-path", model,
        "--port", str(args.port),
        "--gpu", str(args.gpu),
        "--max-running-requests", str(conc),
        "--memory-ratio", f"{ratio:.3f}",
        "--max-prefill-length", str(prefill),
        "--cache-type", cache,
    ]
    if tier == "aggressive":
        cmd += ["--cuda-graph-max-bs", "1"]
    if args.seq_len:
        cmd += ["--max-seq-len-override", str(args.seq_len)]
    if args.extra:
        cmd += args.extra.split()

    print("-" * 70)
    print("  " + " ".join(_q(c) for c in cmd))
    print("=" * 70)
    print()

    if args.dry_run:
        return 0

    log_path = args.log or os.path.join(HERE, "serve.log")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, bufsize=0,
    )
    logf = open(log_path, "w", encoding="utf-8", errors="replace")

    print(f"  日志    : {log_path}（首次加载要捕获 CUDA graph，1~5 分钟正常）")
    print()

    base = f"http://127.0.0.1:{args.port}"
    t_start = time.time()
    next_probe = t_start
    ready = False
    tail: list[str] = []
    try:
        while True:
            chunk = proc.stdout.read(4096) if proc.stdout else b""
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            for seg in _split(chunk):
                line = seg.rstrip()
                if not line:
                    continue
                if _is_progress(line):
                    sys.stdout.write("\r" + line[:110].ljust(110))
                    sys.stdout.flush()
                else:
                    print(line)
                tail.append(line)
                del tail[:-40]
                logf.write(line + "\n")
                logf.flush()

            if not ready and not args.no_wait and time.time() - t_start < 900:
                if time.time() >= next_probe:
                    next_probe = time.time() + 8
                    if _probe(base):
                        ready = True
                        break
    except KeyboardInterrupt:
        print("\n[停止中] 收到 Ctrl+C …")
        _kill(proc.pid)
        logf.close()
        return 0

    code = proc.wait()
    logf.close()

    if ready:
        print()
        print("=" * 66)
        print("  ✓ 服务已就绪")
        print("=" * 66)
        print(f"  OpenAI 兼容接口 : {base}/v1/chat/completions")
        print(f"  模型列表        : {base}/v1/models")
        print(f"  模型名          : {name}")
        print()
        print("  快速自检（新开一个窗口执行）：")
        print(f'    curl {base}/v1/chat/completions -H "Content-Type: application/json" \\')
        print(f'      -d "{{\\"model\\":\\"{name}\\",\\"messages\\":[{{\\"role\\":\\"user\\",'
              f'\\"content\\":\\"用一句话介绍你自己。\\"}}],\\"max_tokens\\":200}}"')
        print()
        print("  也把模型名填进工作网页的「服务与测试」页就能直接对话。")
        print("=" * 66)
        print("  按 Ctrl+C 停止服务。")
        try:
            while proc.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[停止中] …")
            _kill(proc.pid)
        return 0

    print()
    print("=" * 66)
    print(f"  ✗ 服务未就绪就退出了（exit={code}）")
    print("=" * 66)
    err = "\n".join(l for l in tail if re.search(
        r"Error|错误|Assertion|not enough|No such|Failed|Traceback", l, re.I))
    if err:
        print(err[-1500:])
    print()
    print("  排查建议：")
    print("   * Not enough memory for KV cache → 提高 --ratio 或降低 --prefill/--concurrency")
    print("   * 输出乱码 → 权重或内核路径问题，不是参数问题，别继续调参")
    return 1


def _q(s: str) -> str:
    return f'"{s}"' if re.search(r"[\s&()]", s) else s


def _probe(base: str) -> bool:
    """HTTP 端口起来得比模型早得多：/v1/models 立刻就是 200，但引擎还在加载。

    真正就绪的标志是 /v1/chat/completions 能返回 200（加载期间它回 503
    "model is still loading"）。所以这里发一个 1 token 的探针请求。
    """
    body = json.dumps({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def _kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            pass


def _split(buf: bytes) -> list[str]:
    out = []
    for seg in re.split(rb"[\r\n]", buf):
        if not seg:
            continue
        out.append(_decode(seg))
    return out


def _decode(seg: bytes) -> str:
    if b"\x00" in seg:
        try:
            return seg.decode("utf-16-le", errors="replace").replace("\x00", "")
        except Exception:
            pass
    for enc in ("utf-8", "gbk"):
        try:
            return seg.decode(enc)
        except UnicodeDecodeError:
            continue
    return seg.decode("utf-8", errors="replace")


def _is_progress(line: str) -> bool:
    return bool(re.search(r"\d{1,3}(?:\.\d+)?\s*%", line)) and bool(
        re.search(r"[█▉▊▋▌▍▎▏]|\[00:|\d+(?:\.\d+)?\s*[GMK]?B?/s", line))


if __name__ == "__main__":
    sys.exit(main())
