#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""FTW 输出诊断：直接看引擎吐出的 token id，绕开 HTTP 层。

为什么需要它：OpenAI 兼容接口拒绝 `echo` / `logprobs` / `return_token_ids`，
`ft ctl generate` 也只回 {"text": "..."}。所以「返回空串」到底是没出词、
还是出的是乱码 token 被解码器吞掉，只能从引擎的离线 API 看 `output_ids`。

    python diag_tokens.py                    # 自动找模型
    python diag_tokens.py --model G:\AI\xxx-FTW
    python diag_tokens.py --greedy-tokens 32 # 看更多 token

注意：离线模式会**独占约 20GB 显存**，跑之前先停掉正在运行的 ft serve。
必须用引擎自带的 Python：
    %LOCALAPPDATA%\FreeToken\venv\Scripts\python.exe diag_tokens.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROMPT = "The capital of France is"


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"


def die(msg: str, code: int = 1) -> None:
    print(f"[x] {msg}", file=sys.stderr)
    sys.exit(code)


def find_model(explicit: str | None) -> str:
    if explicit:
        if os.path.isdir(explicit):
            return os.path.abspath(explicit)
        die(f"目录不存在：{explicit}")
    env = os.environ.get("FTW_MODEL")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here), r"G:\AI", r"N:\AI"):
        if not os.path.isdir(base):
            continue
        try:
            for n in sorted(os.listdir(base)):
                p = os.path.join(base, n)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "freetoken_weight.json")):
                    return p
        except OSError:
            continue
    die("找不到 FTW 模型目录，请用 --model 指定。")


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == "0":
                return int(float(parts[1]))
    except Exception:
        pass
    return None


def check_ft_installed() -> None:
    try:
        import freetoken  # noqa: F401
    except ImportError:
        die(
            "当前 Python 里没有 freetoken 包。\n"
            r"    请改用：%LOCALAPPDATA%\FreeToken\venv\Scripts\python.exe diag_tokens.py"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="FTW 输出诊断（token id 级）")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--greedy-tokens", type=int, default=16)
    ap.add_argument("--sample-tokens", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.957)
    ap.add_argument("--show", type=int, default=16, help="每个样本打印前 N 个 token 的逐条解码")
    args = ap.parse_args()

    check_ft_installed()

    free = gpu_free_mib()
    if free is not None and free < 20000:
        print(f"[!] GPU 可用显存只有 {free} MiB，离线加载需要 ~20GB。")
        print("    先停掉正在跑的 ft serve / FreeToken Desktop 再试。\n")

    model = find_model(args.model)

    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    print("=" * 72)
    print("  FTW 输出诊断")
    print("=" * 72)
    print(f"  模型 : {model}")
    print(f"  提示 : {args.prompt!r}")
    print("=" * 72)
    print("[*] 离线加载中（会独占显存，约 1~2 分钟）…", flush=True)

    llm = LLM(
        model_path=model,
        dtype=torch.bfloat16,
        memory_ratio=args.ratio,
        max_running_req=1,
        cache_type="naive",
        max_extend_tokens=2048,
        nvfp4_backend="triton",
    )
    tok = llm.tokenizer
    print(f"[*] 加载完成 · eos ids = {sorted(llm.eos_token_ids)} · eos = {tok.eos_token!r}",
          flush=True)

    def run(tag: str, sp) -> None:
        print(f"\n{'─' * 72}\n[{tag}]", flush=True)
        res = llm.generate([args.prompt], sp)[0]
        ids = res["token_ids"]
        print(f"  token 数 = {len(ids)}")
        print(f"  token_ids = {ids[:64]}{' …' if len(ids) > 64 else ''}")
        print(f"  text      = {res['text']!r}")
        for i, t in enumerate(ids[: args.show]):
            piece = tok.decode([t])
            flag = "  <-- 非法/残字节" if "\ufffd" in piece else ""
            print(f"    #{i:<3} id={t:<8} piece={piece!r}{flag}")
        if ids:
            bad = sum(1 for t in ids if "\ufffd" in tok.decode([t]))
            print(f"  → 含残字节的 token：{bad}/{len(ids)}")

    run("A 贪心 temperature=0（走 torch.argmax）",
        SamplingParams(temperature=0.0, max_tokens=args.greedy_tokens, ignore_eos=True))
    run("B 采样 temperature=1.0 / top_k=20 / top_p=0.95（模型推荐值，走 Triton 采样内核）",
        SamplingParams(temperature=1.0, top_k=20, top_p=0.95,
                       max_tokens=args.sample_tokens, ignore_eos=True))

    print(f"\n{'=' * 72}")
    print("  怎么看结果")
    print("=" * 72)
    print("""
  * 第 0 个 token 就该是一个正常词（如 ' Paris'）。
    若它是 '··'、'···' 这类残字节 → logits 从一开始就是坏的，与采样参数无关。
  * 若 token 全是跨语种/代码标识符大杂烩（sourceMapping、:UIControl、燕、稿件来源）
    → 典型的权重或内核数值路径错误。
  * 若 text 为空但 token_ids 正常 → 那才是解码器的问题，不是模型的问题。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
