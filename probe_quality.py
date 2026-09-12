"""输出质量探针：对已运行的 ft serve 实例发确定性(greedy)请求，判断输出是否连贯。

用法:
    python probe_quality.py --port 8421
    python probe_quality.py --port 8451 --launch --env FREETOKEN_DEBUG_FP8_REF=1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

FT = r"C:\Users\zymzy\AppData\Local\FreeToken\venv\Scripts\ft.exe"
MODEL = r"G:\AI\Qwen3.8-27B-NVFP4-FTW"
SERVED = "Qwen3.8-27B-NVFP4-FTW"

# 最容易被模型接对的两个提示，greedy 下应当稳定输出连贯英文
PROBES = [
    ("raw-completion", "/v1/completions", {
        "model": SERVED,
        "prompt": "The capital of France is",
        "max_tokens": 24, "temperature": 0,
    }),
    ("chat", "/v1/chat/completions", {
        "model": SERVED,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
        "max_tokens": 32, "temperature": 0,
    }),
]


def http_json(port: int, path: str, payload=None, timeout: float = 300.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def wait_ready(port: int, timeout: float = 240.0) -> None:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            h = http_json(port, "/health", timeout=5)
            st = h.get("state") or h.get("status") or "?"
            if st != last:
                print(f"  [health] {st} {h.get('progress', '')}", flush=True)
                last = st
            if st in ("ok", "ready"):
                return
            if st in ("error", "failed"):
                raise SystemExit(f"服务启动失败: {h}")
        except urllib.error.URLError:
            pass
        time.sleep(2)
    raise SystemExit("等待服务就绪超时")


def extract(resp: dict) -> str:
    try:
        if "choices" in resp:
            ch = resp["choices"][0]
            if "text" in ch:
                return ch["text"]
            return (ch.get("message") or {}).get("content") or ""
    except Exception:  # noqa: BLE001
        pass
    return json.dumps(resp, ensure_ascii=False)[:400]


def run_probes(port: int) -> list[dict]:
    out = []
    for name, path, payload in PROBES:
        t0 = time.time()
        try:
            resp = http_json(port, path, payload)
            text = extract(resp)
            usage = resp.get("usage") or {}
            out.append({"name": name, "ok": True, "text": text,
                        "sec": round(time.time() - t0, 1), "usage": usage})
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "ok": False, "text": f"<ERROR {e}>",
                        "sec": round(time.time() - t0, 1), "usage": {}})
    return out


def report(results: list[dict]) -> None:
    print("\n" + "=" * 68)
    for r in results:
        t = r["text"].replace("\n", "\\n")
        print(f"[{r['name']}] {r['sec']}s  {r['usage']}")
        print(f"   -> {t[:300]}")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--launch", action="store_true", help="自行启动 ft serve")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL，可重复")
    ap.add_argument("--arg", action="append", default=[], help="额外 ft serve 参数，可重复")
    ap.add_argument("--memory-ratio", default="0.956")
    args = ap.parse_args()

    proc = None
    if args.launch:
        env = os.environ.copy()
        for kv in args.env:
            k, _, v = kv.partition("=")
            env[k.strip()] = v.strip()
            print(f"  env {k.strip()}={v.strip()}", flush=True)
        cmd = [FT, "serve", "--model-path", MODEL, "--port", str(args.port),
               "--max-running-requests", "1", "--memory-ratio", args.memory_ratio,
               "--max-prefill-length", "4096", "--cache-type", "naive", *args.arg]
        print("启动:", " ".join(cmd), flush=True)
        logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_server.log")
        logf = open(logpath, "wb", buffering=0)
        print(f"  服务日志 -> {logpath}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, bufsize=0)
        wait_ready(args.port)

    try:
        report(run_probes(args.port))
    finally:
        if proc is not None:
            print("关闭服务 ...", flush=True)
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
