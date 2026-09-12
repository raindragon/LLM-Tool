#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FreeToken 转换工具 - 本地后端服务

只依赖 Python 标准库，无需安装任何第三方包。

    python server.py [--port 8420] [--no-browser]

功能：
  * 环境探测   ft.exe / GPU / 磁盘
  * 目录浏览   带权重的目录高亮
  * 格式转换   ft checkpoint  -> FTW
  * 模型服务   ft serve       -> OpenAI 兼容 API
  * 在线测试   直接调用 /v1/chat/completions 验证生成质量
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".ftw", ".pth")
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0

# --------------------------------------------------------------------------- #
# 全局状态
# --------------------------------------------------------------------------- #

TASKS: dict[str, "Task"] = {}
TASKS_LOCK = threading.Lock()


class Task:
    """一个被托管的子进程 + 它到目前为止的全部输出。"""

    def __init__(self, kind: str, cmd: list[str], meta: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                      # convert | serve
        self.cmd = cmd
        self.meta = meta or {}
        self.lines: list[str] = []
        self.offset = 0                       # 客户端已读取到的行号
        self.status = "running"               # running | done | error | cancelled
        self.exit_code: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.progress = {"percent": None, "speed": None, "detail": None}
        self.phase = "开始"                     # 启动阶段的粗粒度状态
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    # -- 输出 ------------------------------------------------------------- #
    def add(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > 20000:       # 防止超长任务吃光内存
                drop = len(self.lines) - 20000
                self.lines = self.lines[drop:]
                self.offset = max(0, self.offset - drop)
        self._parse(line)

    def _parse(self, line: str) -> None:
        """从进度条输出里提取百分比 / 速度，并推断当前所处阶段。"""
        m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", line)
        if m:
            try:
                self.progress["percent"] = min(100.0, float(m.group(1)))
            except ValueError:
                pass
        m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)/s", line)
        if m:
            self.progress["speed"] = f"{m.group(1)} {m.group(2)}/s"
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(it/s|s/it|tok/s|token/s)", line)
            if m:
                self.progress["speed"] = f"{m.group(1)} {m.group(2)}"
        m = re.search(r"(\d+(?:\.\d+)?\s*[KMGT]?B?)\s*/\s*(\d+(?:\.\d+)?\s*[KMGT]?B?)", line)
        if m:
            self.progress["detail"] = f"{m.group(1).strip()} / {m.group(2).strip()}"
        m = re.search(r"\[(\d+:\d+(?::\d+)?)<([\d:]+)", line)
        if m:
            self.progress["detail"] = self.progress.get("detail") or f"剩余 {m.group(2)}"
        self._phase(line)

    # 阶段的先后顺序（数字越大越靠后）。不按列表顺序排，是因为两类引擎的日志
    # 交错方式不同，硬排容易把"就绪"排到"HTTP 已监听"前面。
    PHASE_RANK = {
        "开始": 0,
        "创建运行环境": 1,
        "读取模型": 1,
        "加载模型": 1,
        "加载权重": 2,
        "上传权重到显存": 3,
        "安装依赖": 4,
        "初始化上下文": 4,
        "分配 KV 缓存": 5,
        "准备捕获图": 6,
        "捕获 CUDA graph": 7,
        "预热": 8,
        "模型已加载": 9,
        "HTTP 已监听": 9,
        "就绪": 10,
        "安装完成": 10,
    }
    PHASE_ERR = {"显存不足", "启动失败", "权重不完整"}

    # 标志行 -> 阶段。**列表顺序 = 匹配优先级**（同一行可能命中多个标记）。
    # 两处特别注意：
    #   * 错误标记放最前，一旦命中就锁死；
    #   * llama.cpp 把 "server is listening … starting the main loop" 打在同一行，
    #     所以「就绪」必须排在「HTTP 已监听」前面。
    PHASE_MARKS = (
        ("Not enough memory for KV cache", "显存不足"),
        ("Incomplete projection fusions", "权重不完整"),
        ("Traceback (most recent call last)", "启动失败"),
        ("failed to load model", "启动失败"),
        ("error loading model", "启动失败"),
        ("error loading model from", "启动失败"),
        ("unable to load model", "启动失败"),
        ("exiting due to", "启动失败"),
        ("model loading failed", "启动失败"),
        ("starting the main loop", "就绪"),
        ("all slots are idle", "就绪"),
        ("is ready to serve", "就绪"),
        ("API ready", "就绪"),
        ("server is listening", "HTTP 已监听"),
        ("Started server process", "HTTP 已监听"),
        ("Uvicorn running", "HTTP 已监听"),
        ("warming up the model", "预热"),
        ("Prefill warmup", "预热"),
        # 新版 llama-server 只说 "listening on http://…"
        # 老版说的是 "main: server is listening on … - starting the main loop"
        # 两种都算就绪，所以「listening on http://」必须排在「HTTP 已监听」前面。
        ("listening on http://", "就绪"),
        ("llama_server: model loaded", "模型已加载"),
        ("Starting installation", "创建运行环境"),
        ("creating venv", "创建运行环境"),
        ("Installing freetoken", "安装依赖"),
        ("Successfully installed", "安装完成"),
        ("ft installed at", "安装完成"),
        ("install complete", "安装完成"),
        ("Preparing for capturing CUDA graphs", "准备捕获图"),
        ("Start capturing CUDA graphs", "捕获 CUDA graph"),
        ("Capturing graphs", "捕获 CUDA graph"),
        ("llama_kv_cache", "分配 KV 缓存"),
        ("llama_context", "初始化上下文"),
        ("load_tensors", "上传权重到显存"),
        ("Loading weights", "加载权重"),
        ("Loading model", "加载模型"),
        ("llama_model_loader", "读取模型"),
    )

    def _phase(self, line: str) -> None:
        for mark, phase in self.PHASE_MARKS:
            if mark not in line:
                continue
            if phase in self.PHASE_ERR:
                self.phase = phase                    # 错误一票通过，且不可被覆盖
            elif self.phase not in self.PHASE_ERR and \
                    self.PHASE_RANK.get(phase, 0) >= self.PHASE_RANK.get(self.phase, 0):
                self.phase = phase                    # 正常阶段只允许前进
            return

    # -- 快照 ------------------------------------------------------------- #
    def snapshot(self, from_offset: int) -> dict:
        with self.lock:
            end = len(self.lines)
            start = max(0, min(from_offset, end))
            new = self.lines[start:end]
            total = end
        elapsed = (self.finished or time.time()) - self.started
        pct = self.progress.get("percent")
        eta = None
        if self.status == "running" and pct and pct > 0.5:
            eta = max(0.0, elapsed * (100.0 - pct) / pct)
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "exitCode": self.exit_code,
            "lines": new,
            "offset": total,
            "elapsed": round(elapsed, 1),
            "eta": round(eta, 1) if eta is not None else None,
            "progress": dict(self.progress),
            "phase": self.phase,
            "pid": self.proc.pid if self.proc else None,
            "meta": self.meta,
        }


def _spawn(task: Task) -> None:
    env = dict(os.environ)
    # 让子进程用 UTF-8 输出，否则中文 Windows 上会按 cp936/UTF-16 混杂输出
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONUTF8", "1")

    try:
        task.proc = subprocess.Popen(
            task.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # bufsize=0 很关键：默认的 BufferedReader.read(n) 会一直阻塞到凑满 n 字节，
            # 引擎打印完 "API ready" 之后长时间不再说话，日志就会永远卡在最后一行不刷新。
            bufsize=0,
            cwd=task.meta.get("cwd") or os.path.dirname(task.cmd[0]) or None,
            env=env,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        task.add(f"[启动失败] {exc}")
        task.status = "error"
        task.exit_code = -1
        task.finished = time.time()
        return

    stream = task.proc.stdout
    buf = b""
    assert stream is not None
    while True:
        try:
            chunk = stream.read(8192)      # bufsize=0 -> 返回当前可读的全部字节
        except Exception:
            break
        if not chunk:
            break
        buf += chunk
        # ft 的进度条用 \r 刷新，两种换行都要切
        while True:
            idx = -1
            for sep in (b"\n", b"\r"):
                i = buf.find(sep)
                if i != -1 and (idx == -1 or i < idx):
                    idx = i
            if idx == -1:
                break
            seg, buf = buf[:idx], buf[idx + 1:]
            task.add(_decode(seg))
    if buf:
        task.add(_decode(buf))

    code = task.proc.wait()
    with TASKS_LOCK:
        if task.status == "running":
            task.status = "done" if code == 0 else "error"
    task.exit_code = code
    if task.finished is None:
        task.finished = time.time()
    if task.status == "done" and task.progress.get("percent") is None:
        task.progress["percent"] = 100.0


def _decode(seg: bytes) -> str:
    """兼容 UTF-8 / UTF-16LE / 本地代码页，返回可读文本。"""
    if not seg:
        return ""
    if seg[:1] == b"\x00":
        seg = seg.lstrip(b"\x00")            # UTF-16 残留
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
    return seg.decode("utf-8", errors="replace").replace("\x00", "")


def _start_task(kind: str, cmd: list[str], meta: dict | None = None) -> Task:
    task = Task(kind, cmd, meta)
    with TASKS_LOCK:
        TASKS[task.id] = task
    threading.Thread(target=_spawn, args=(task,), daemon=True).start()
    return task


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), 15)
        except Exception:
            pass


def _shutdown_children() -> None:
    """工具退出时顺手把托管的服务收掉。

    模型服务会占掉 20+ GB 显存，留成孤儿进程最难排查（GPU 被占着却找不到是谁）。
    所以在退出路径上主动清理，"关掉工具 = 显存归还"。
    """
    with TASKS_LOCK:
        running = [t for t in TASKS.values() if t.status == "running" and t.proc]
    for t in running:
        try:
            print(f"  正在停止 {t.kind} 任务 {t.id}（pid {t.proc.pid}）…", flush=True)
            _kill_tree(t.proc.pid)
        except Exception:
            pass


def _find_task_by_port(port) -> "Task | None":
    """找一个占用该端口、且由本工具托管、仍在跑的任务。"""
    port = str(port or "")
    with TASKS_LOCK:
        for t in TASKS.values():
            if (t.status == "running" and t.proc
                    and str((t.meta or {}).get("port") or "") == port):
                return t
    return None


def _stop_existing_on_port(port, label: str) -> tuple[bool, str | None]:
    """端口被占用时：若占用者是我们托管的服务，先停掉再起；否则报冲突。

    返回 (ok, err_msg)。ok=False 时调用方应当直接拒绝启动。
    """
    existing = _find_task_by_port(port)
    if existing:
        try:
            _kill_tree(existing.proc.pid)
            existing.status = "cancelled"
            existing.finished = time.time()
            existing.add(f"[已被新启动请求接管] 旧 {label} 任务 {existing.id} 已停止")
        except Exception:
            pass
        # 等端口真正空出来（llama-server 退场要一两秒）
        for _ in range(30):
            if not port_listening(port):
                break
            time.sleep(0.5)
        return True, None
    return False, (f"端口 {port} 已经被一个不是本工具启动的服务占用。"
                   f"请先停掉它，或换个端口。")


# --------------------------------------------------------------------------- #
# 环境探测
# --------------------------------------------------------------------------- #

# nvidia-smi 常常没在 PATH 里（尤其在中文 Windows / 精简环境），但它几乎一定装在
# NVIDIA 驱动自带的 NVSMI 目录，或新版驱动的 System32 根下。
#
# ⚠ 关键坑：32 位解释器（`py -3` 有时会挑到 "Python311-32"）跑在 64 位 Windows 上时，
#   System32 会被 WOW64 **静默重定向**到 SysWOW64 —— 而那里既没有 nvidia-smi.exe、
#   也没有 nvml.dll（已实测：两个文件在 SysWOW64 下都不存在），于是出现
#   「机器有显卡、nvidia-smi 在命令行里能跑，但工具完全探测不到」的诡异现象。
#   32 位进程想访问真正的 System32，唯一通道是 `C:\Windows\Sysnative` 这个虚拟别名。
_NVSMI_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Windows\Sysnative\nvidia-smi.exe",
    r"C:\Windows\SysWOW64\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi",
]

# NVML 兜底库。32 位进程加载不了 64 位的 System32\nvml.dll（位数不匹配），
# 所以 32 位下必须靠 nvidia-smi；这里照着列，交给 CDLL 自己挑能装的那个。
_NVML_CANDIDATES = [
    "nvml.dll",
    r"C:\Windows\System32\nvml.dll",
    r"C:\Windows\Sysnative\nvml.dll",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
]

_SMI_PATH_CACHE: list[str | None] = [None]


def _is_wow64() -> bool:
    """32 位解释器跑在 64 位 Windows 上（此时 System32 被重定向到 SysWOW64）。"""
    if os.name != "nt" or sys.maxsize > 2 ** 32:
        return False
    try:
        flag = ctypes.c_int(0)
        if ctypes.windll.kernel32.IsWow64Process(ctypes.c_void_p(-1), ctypes.byref(flag)):
            return bool(flag.value)
    except Exception:
        pass
    return True  # 真 32 位 Windows 上 Sysnative 不存在，多试一次没有副作用


def _host_bits() -> int:
    return 64 if sys.maxsize > 2 ** 32 else 32


def _load_nvml():
    """按候选表加载 nvml.dll；都失败返回 None。"""
    for c in _NVML_CANDIDATES:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def _smi_candidates() -> list[str]:
    """候选 nvidia-smi 路径，按「最可能可用」排序并去重。"""
    raw: list[str | None] = []
    if _is_wow64():
        raw.append(r"C:\Windows\Sysnative\nvidia-smi.exe")  # 32 位进程唯一能用的通道
    raw += [shutil.which("nvidia-smi"), shutil.which("nvidia-smi.exe")]
    raw += _NVSMI_CANDIDATES
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        if not c or c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    return out


def _nvidia_smi_path() -> str | None:
    """返回第一个**真实存在且能跑出显卡名**的 nvidia-smi（进程内永久缓存）。

    不能只看 ``os.path.isfile`` / ``shutil.which``：32 位进程下 System32 的重定向会
    让它们给出假结果（实测 which 返回 None、isfile 返回 False），所以这里真跑一次
    ``--query-gpu=name`` 来验证，选中的路径缓存下来，后续调用零开销。
    """
    if _SMI_PATH_CACHE[0]:
        return _SMI_PATH_CACHE[0]
    for c in _smi_candidates():
        try:
            if not os.path.isfile(c):
                continue
        except OSError:
            continue
        try:
            r = subprocess.run([c, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15,
                               creationflags=CREATE_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                _SMI_PATH_CACHE[0] = c
                return c
        except Exception:
            continue
    return None


def _gpu_list_nvml() -> list[dict]:
    """nvidia-smi 彻底找不到时的兜底：直接调驱动自带的 NVML（nvml.dll）。"""
    try:
        nvml = ctypes.CDLL("nvml.dll")
    except OSError:
        try:
            nvml = ctypes.CDLL(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll")
        except OSError:
            return []

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_uint64), ("free", ctypes.c_uint64), ("used", ctypes.c_uint64)]

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint32), ("memory", ctypes.c_uint32)]

    # NVML 1.x 结构体（足够拿到显存 / 利用率 / 温度 / 功耗）
    class _Pci(ctypes.Structure):
        _fields_ = [("busId", ctypes.c_char * 16), ("domain", ctypes.c_uint32),
                    ("bus", ctypes.c_uint32), ("device", ctypes.c_uint32)]

    name_buf = ctypes.create_string_buffer(128)
    mem = _Mem()
    util = _Util()
    temp = ctypes.c_uint32()
    power = ctypes.c_uint32()
    gpus: list[dict] = []

    try:
        if nvml.nvmlInit_v2() != 0:
            return []
        count = ctypes.c_uint32(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            count.value = 0
        for i in range(count.value):
            h = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                continue
            nvml.nvmlDeviceGetName(h, name_buf, ctypes.c_uint32(128))
            try:
                nm = name_buf.value.decode("utf-8", "replace").strip()
            except Exception:
                nm = f"GPU {i}"
            used = 0
            if nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0 and mem.total:
                used = int(mem.total - mem.free)
                free = int(mem.free)
                total = int(mem.total)
            else:
                free = total = 0
            gpu_util = temp_c = power_w = 0
            if nvml.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == 0:
                gpu_util = int(util.gpu)
            try:
                if nvml.nvmlDeviceGetTemperature(h, ctypes.c_uint32(0), ctypes.byref(temp)) == 0:
                    temp_c = int(temp.value)
            except Exception:
                pass
            try:
                if nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(power)) == 0:
                    power_w = round(power.value / 1000.0, 1)   # 毫瓦 → 瓦
            except Exception:
                pass
            gpus.append({
                "index": str(i), "name": nm,
                "totalMiB": total and total // (1024 * 1024),
                "usedMiB": used // (1024 * 1024),
                "freeMiB": free // (1024 * 1024),
                "computeCap": "", "source": "nvml",
                "util": gpu_util or None,
                "tempC": temp_c or None,
                "powerW": power_w or None,
                "powerLimitW": None,
            })
    except Exception:
        gpus = []
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def _gpu_list_impl() -> list[dict]:
    smi = _nvidia_smi_path()
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
                creationflags=CREATE_NO_WINDOW,
            )
            gpus = []
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": parts[0], "name": parts[1],
                        "totalMiB": int(float(parts[2])), "usedMiB": int(float(parts[3])),
                        "freeMiB": int(float(parts[4])), "computeCap": parts[5],
                        "source": "nvidia-smi",
                    })
            if gpus:
                return gpus
        except Exception:
            pass
    # 回退：UUID 列表（拿不到显存时至少认出卡）
    if smi:
        try:
            out = subprocess.run([smi, "-L"], capture_output=True, text=True,
                                 timeout=15, creationflags=CREATE_NO_WINDOW)
            rows = [{"index": str(i), "name": ln.split(":", 1)[-1].split("(")[0].strip(),
                     "uuid": ln.split("UUID:")[-1].rstrip(") \n") if "UUID:" in ln else None,
                     "source": "nvidia-smi"}
                    for i, ln in enumerate(l for l in out.stdout.strip().splitlines() if l.strip())]
            if rows:
                return rows
        except Exception:
            pass
    # 最后兜底：NVML（无需 nvidia-smi 即可读显存）
    nvml_gpus = _gpu_list_nvml()
    if nvml_gpus:
        return nvml_gpus
    return []


def gpu_list() -> list[dict]:
    """带缓存的显卡列表 —— 同一进程内 1.5 秒内多次调用只跑一次 nvidia-smi / NVML。"""
    return _smi_cached("list", _gpu_list_impl)


def drive_list() -> list[dict]:
    out = []
    for d in string.ascii_uppercase:
        root = f"{d}:\\"
        if os.path.exists(root):
            try:
                t, u, f = shutil.disk_usage(root)
                out.append({"drive": d, "totalGB": round(t / 2**30, 1), "freeGB": round(f / 2**30, 1)})
            except OSError:
                pass
    return out


def is_model_dir(path: str) -> dict:
    """判断目录里有没有可转换的权重，并统计大小。"""
    info = {"safetensors": 0, "gguf": 0, "ftw": False, "shards": 0, "other": 0, "bytes": 0,
            "hasConfig": False, "quant": None}
    try:
        for entry in os.scandir(path):
            if not entry.is_file():
                continue
            low = entry.name.lower()
            if low.endswith(".safetensors"):
                info["safetensors"] += 1
                try:
                    info["bytes"] += entry.stat().st_size
                except OSError:
                    pass
            elif low.endswith(".gguf"):
                info["gguf"] += 1
                try:
                    info["bytes"] += entry.stat().st_size
                except OSError:
                    pass
            elif low == "freetoken_weight.json":
                info["ftw"] = True
            elif low.endswith(".ftw"):
                info["shards"] = info.get("shards", 0) + 1
                try:
                    info["bytes"] += entry.stat().st_size
                except OSError:
                    pass
            elif low == "config.json":
                info["hasConfig"] = True
            elif low.endswith(WEIGHT_EXT):
                info["other"] += 1
    except OSError as exc:
        info["error"] = str(exc)
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            info["arch"] = (j.get("architectures") or [None])[0] or j.get("model_type")
            tc = j.get("text_config") or {}
            info["layers"] = tc.get("num_hidden_layers") or j.get("num_hidden_layers")
        except Exception:
            pass
    return info


def scan_models(root: str, depth: int = 2) -> list[dict]:
    root = os.path.abspath(root)
    found, seen = [], set()
    base_depth = root.rstrip("\\/").count(os.sep)
    for cur, dirs, _files in os.walk(root):
        if cur.count(os.sep) - base_depth > depth:
            dirs[:] = []
            continue
        if os.sep + "." in cur:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        info = is_model_dir(cur)
        if (info["safetensors"] or info["gguf"] or info["ftw"]) and cur not in seen:
            seen.add(cur)
            found.append({"path": cur, "name": os.path.basename(cur) or cur, **info})
    found.sort(key=lambda m: (not m["ftw"], -m["bytes"]))
    return found


# --------------------------------------------------------------------------- #
# llama.cpp / llama-turboquant
# --------------------------------------------------------------------------- #
#
# 两个引擎的二进制都叫 llama-server.exe，参数集几乎完全一致 —— 实测把两份
# ``--help`` 拉平对比，只有 turboquant 定制版多出 turbo2/3/4 这几种压缩 KV。
# 所以这里共用一套命令行构造器，用 ``--help`` 探测能力来决定能不能用 turbo KV。
#
# 参考：https://github.com/ggml-org/llama.cpp
#       https://github.com/TheTom/llama-cpp-turboquant

LLAMA_EXE = "llama-server.exe"

# 探测安装位置的根目录（每个只往下找 4 层，不做全盘扫描）
LLAMA_ROOTS = [
    r"G:\AI", r"G:\AI\LammacppStart", r"C:\AI", r"D:\AI", r"E:\AI",
    r"C:\llama.cpp", r"C:\Program Files\llama.cpp", r"D:\llama.cpp",
]

# 启动档位。turbo 档只在支持 turbo KV 的构建上出现，前端会按能力过滤。
LLAMA_PRESETS = [
    {"id": "turbo128k", "label": "128K 极致容量（参考脚本）", "ctx": 131072, "kvK": "turbo2", "kvV": "turbo2",
     "batch": 256, "ubatch": 512, "cacheRam": 16384, "turbo": True,
     "threads": 8, "threadsBatch": 12, "parallel": 1, "ngl": 99,
     "loadMode": "mmap", "reasoning": "off", "warmup": "1",
     "logitBias": "248066-inf,248067-inf", "flashAttn": "on",
     "note": "对齐 start - 128K极致容量版.bat：turbo2/turbo2 压缩 KV、128K 上下文、关思考链、预热（长文档首选，速度偏慢）"},
    {"id": "balanced", "label": "均衡（推荐）", "ctx": 65536, "kvK": "turbo2", "kvV": "turbo2",
     "batch": 512, "ubatch": 1024, "cacheRam": 2048, "turbo": True,
     "note": "turbo2 压缩 KV，速度与容量兼顾（对应 start-server.bat 的默认档）"},
    {"id": "quality", "label": "质量优先", "ctx": 24576, "kvK": "turbo3", "kvV": "turbo3",
     "batch": 1024, "ubatch": 2048, "cacheRam": 1024, "turbo": True,
     "note": "压缩最轻，输出质量最接近原模型，上下文偏短"},
    {"id": "long", "label": "长上下文", "ctx": 131072, "kvK": "turbo2", "kvV": "turbo2",
     "batch": 256, "ubatch": 512, "cacheRam": 16384, "turbo": True,
     "note": "吃满 128K 上下文，速度明显变慢"},
    {"id": "std", "label": "标准 KV（f16 / q8_0）", "ctx": 32768, "kvK": "f16", "kvV": "q8_0",
     "batch": 2048, "ubatch": 512, "cacheRam": 8192, "turbo": False,
     "note": "不用 turbo 压缩，任何 llama.cpp 构建都能跑"},
]

TURBO_KV = ("turbo2", "turbo3", "turbo4")


def _run_quiet(cmd: list[str], timeout: float = 30, cwd: str | None = None) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd,
                           creationflags=CREATE_NO_WINDOW)
    except Exception:
        return ""
    return _decode((p.stdout or b"") + (p.stderr or b""))


_CAPS_CACHE: dict[str, tuple[float, dict]] = {}


def llama_caps(exe: str) -> dict:
    """探测某个 llama-server.exe 支持哪些参数（按文件 mtime 缓存，避免反复跑 --help）。"""
    if not exe or not os.path.isfile(exe):
        return {}
    try:
        stamp = os.path.getmtime(exe)
    except OSError:
        return {}
    hit = _CAPS_CACHE.get(exe)
    if hit and hit[0] == stamp:
        return hit[1]
    txt = _run_quiet([exe, "--help"], 40)
    caps = {
        "turboKV": bool(re.search(r"\bturbo[234]\b", txt)),
        "mmproj": "--mmproj" in txt,
        "loadMode": "--load-mode" in txt,
        "reasoning": "--reasoning" in txt,
        "logitBias": "--logit-bias" in txt,
        "metrics": "--metrics" in txt,
        "cacheRam": "--cache-ram" in txt,
        "helpChars": len(txt),
    }
    m = re.search(r"version:\s*([^\s(]+)(?:\s*\(build\s+(\d+))?", txt)
    if not m:
        m = re.search(r"build:\s*([0-9a-zA-Z.+-]+)", txt)
    if m:
        ver = m.group(1)
        bnum = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        caps["build"] = "" if ver in ("0", "unknown") else (f"{ver} ({bnum})" if bnum else ver)
    else:
        caps["build"] = ""
    if not caps["build"]:
        # --help 不带版本号，得单独问一次
        vm = re.search(r"version:\s*([^\s(]+)(?:\s*\(build\s+(\d+))?",
                       _run_quiet([exe, "--version"], 25))
        if vm:
            ver = vm.group(1)
            bnum = vm.group(2)
            caps["build"] = "" if ver in ("0", "unknown") else (f"{ver} ({bnum})" if bnum else ver)
    _CAPS_CACHE[exe] = (stamp, caps)
    return caps


def _walk_llama_exe(root: str, max_depth: int = 4) -> list[str]:
    """在 root 下 max_depth 层内找所有 llama-server.exe。"""
    root = os.path.abspath(root)
    base = root.rstrip("\\/").count(os.sep)
    out: list[str] = []
    skip = {"$recycle.bin", "system volume information", "node_modules", ".git", "__pycache__"}
    for cur, dirs, files in os.walk(root):
        if cur.count(os.sep) - base > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in skip]
        for f in files:
            if f.lower() == LLAMA_EXE:
                out.append(os.path.join(cur, f))
    return out


def find_llama_exe(path: str) -> str | None:
    """path 可以是 llama-server.exe 本身，也可以是一个目录（自动往下找）。"""
    if not path:
        return None
    p = os.path.abspath(str(path))
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        return None
    hits = _walk_llama_exe(p, 4)
    if not hits:
        return None
    hits.sort(key=len)
    return hits[0]


def detect_llama_installs(extra_roots: list[str] | None = None) -> list[dict]:
    """扫描常见的 llama.cpp 安装位置，返回去重后的安装列表。"""
    roots: list[str] = []
    seen_root = set()
    for r in list(extra_roots or []) + LLAMA_ROOTS:
        r = (r or "").strip()
        if not r or not os.path.isdir(r):
            continue
        key = os.path.abspath(r).lower()
        if key in seen_root:
            continue
        seen_root.add(key)
        roots.append(r)

    out: list[dict] = []
    seen_dir = set()
    for root in roots:
        for exe in _walk_llama_exe(root, 4):
            d = os.path.dirname(exe)
            key = os.path.abspath(d).lower()
            if key in seen_dir:
                continue
            seen_dir.add(key)
            caps = llama_caps(exe)
            out.append({
                "dir": d,
                "exe": exe,
                "flavor": "turboquant" if caps.get("turboKV") else "standard",
                "label": "llama-turboquant" if caps.get("turboKV") else "llama.cpp",
                "caps": caps,
            })
    # turboquant 排前面：它能力是标准版的超集
    out.sort(key=lambda x: (x["flavor"] != "turboquant", x["dir"].lower()))
    return out


def scan_gguf(root: str, depth: int = 2) -> list[dict]:
    """列出目录下的 .gguf。mmproj 单独标记（它是视觉投影，不是主模型）。"""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []
    base = root.rstrip("\\/").count(os.sep)
    out: list[dict] = []
    for cur, dirs, files in os.walk(root):
        if cur.count(os.sep) - base > depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if not f.lower().endswith(".gguf"):
                continue
            fp = os.path.join(cur, f)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            low = f.lower()
            out.append({
                "path": fp,
                "name": f,
                "dir": cur,
                "sizeBytes": st.st_size,
                "sizeGB": round(st.st_size / GIB, 2),
                "mtime": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                "mmproj": low.startswith("mmproj"),
                "split": bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", low)),
            })
    out.sort(key=lambda x: (x["mmproj"], -x["sizeBytes"]))
    return out


# --------------------------------------------------------------------------- #
# 系统信息（纯标准库：ctypes 取 CPU / 内存，nvidia-smi 取显卡）
# --------------------------------------------------------------------------- #

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem_info() -> dict:
    if os.name != "nt":
        return {}
    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return {}
        return {
            "totalBytes": int(st.ullTotalPhys),
            "usedBytes": int(st.ullTotalPhys - st.ullAvailPhys),
            "availBytes": int(st.ullAvailPhys),
            "percent": float(st.dwMemoryLoad),
        }
    except Exception:
        return {}


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]


def _ft_val(ft: "_FILETIME") -> int:
    return (ft.hi << 32) | ft.lo


_cpu_lock = threading.Lock()
_cpu_prev: tuple[int, int, int] | None = None
_cpu_last: dict = {"percent": None, "ts": 0.0}


# GPU 探测的缓存（避免轮询时多线程同时跑 nvidia-smi 撞 NVML，结果偶发为空）。
# 同 key 在 TTL 秒内只跑一次真实调用；其它线程从缓存读，绝不让 nvidia-smi 撞车。
_SMI_LOCK = threading.Lock()
_SMI_CACHE: dict[str, tuple[float, list]] = {}
_SMI_TTL = 1.5  # 秒


def _smi_cached(key: str, fn) -> list[dict]:
    """对 nvidia-smi / NVML 调用做线程安全 + TTL 缓存；并发也只跑一次底层调用。"""
    now = time.time()
    hit = _SMI_CACHE.get(key)
    if hit and now - hit[0] < _SMI_TTL:
        return hit[1]
    with _SMI_LOCK:
        hit2 = _SMI_CACHE.get(key)
        if hit2 and now - hit2[0] < _SMI_TTL:
            return hit2[1]
        try:
            val = fn()
        except Exception:
            val = []
        _SMI_CACHE[key] = (now, val)
        return val


def _cpu_sample() -> None:
    """采一次 CPU 累计时间。kernel 时间里**含** idle，所以 busy = (kernel+user) − idle。"""
    global _cpu_prev
    if os.name != "nt":
        return
    try:
        idle, kern, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
            return
    except Exception:
        return
    cur = (_ft_val(idle), _ft_val(kern), _ft_val(user))
    with _cpu_lock:
        prev, _cpu_prev = _cpu_prev, cur
        if prev is None:
            return
        d_idle = cur[0] - prev[0]
        d_total = (cur[1] - prev[1]) + (cur[2] - prev[2])
        if d_total <= 0:
            return
        _cpu_last["percent"] = round(max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0)), 1)
        _cpu_last["ts"] = time.time()


def _cpu_loop() -> None:
    """后台每秒采样一次，这样前端第一次取就有真实数字（否则首帧永远是 0）。"""
    while True:
        try:
            _cpu_sample()
        except Exception:
            pass
        time.sleep(1.0)


def cpu_percent():
    with _cpu_lock:
        return _cpu_last["percent"]


def _cpu_reg(key: str):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            v, _ = winreg.QueryValueEx(k, key)
        return v
    except Exception:
        return None


def cpu_name() -> str:
    v = _cpu_reg("ProcessorNameString")
    if v:
        return re.sub(r"\s+", " ", str(v)).strip()
    return os.environ.get("PROCESSOR_IDENTIFIER") or "未知 CPU"


def cpu_mhz():
    v = _cpu_reg("~MHz")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _gpu_metrics_impl() -> list[dict]:
    """nvidia-smi 的利用率 / 温度 / 功耗 —— 环形仪表要用。

    优先用 nvidia-smi（路径自动定位）；拿不到时退回 NVML。
    """
    smi = _nvidia_smi_path()
    if smi:
        try:
            out = subprocess.run(
                [smi,
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW)
            def num(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            rows = []
            for line in out.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) < 8:
                    continue
                rows.append({
                    "index": p[0], "name": p[1],
                    "util": num(p[2]), "usedMiB": num(p[3]), "totalMiB": num(p[4]),
                    "tempC": num(p[5]), "powerW": num(p[6]), "powerLimitW": num(p[7]),
                    "source": "nvidia-smi",
                })
            if rows:
                return rows
        except Exception:
            pass
    return _gpu_metrics_nvml()


def gpu_metrics() -> list[dict]:
    """带缓存的实时显卡指标 —— 与 gpu_list 共享同一个缓存层。"""
    return _smi_cached("metrics", _gpu_metrics_impl)


def _gpu_metrics_nvml() -> list[dict]:
    """用 NVML 读实时利用率 / 温度 / 功耗（nvidia-smi 不可用时的兜底）。"""
    nvml = _load_nvml()
    if nvml is None:
        return []

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_uint64), ("free", ctypes.c_uint64), ("used", ctypes.c_uint64)]

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint32), ("memory", ctypes.c_uint32)]

    name_buf = ctypes.create_string_buffer(128)
    mem = _Mem()
    util = _Util()
    temp = ctypes.c_uint32()
    power = ctypes.c_uint32()
    rows: list[dict] = []
    try:
        if nvml.nvmlInit_v2() != 0:
            return []
        count = ctypes.c_uint32(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            count.value = 0
        for i in range(count.value):
            h = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                continue
            nvml.nvmlDeviceGetName(h, name_buf, ctypes.c_uint32(128))
            try:
                nm = name_buf.value.decode("utf-8", "replace").strip()
            except Exception:
                nm = f"GPU {i}"
            used = total = 0
            if nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0 and mem.total:
                used = int(mem.total - mem.free)
                total = int(mem.total)
            gpu_util = temp_c = power_w = 0
            if nvml.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == 0:
                gpu_util = int(util.gpu)
            try:
                if nvml.nvmlDeviceGetTemperature(h, ctypes.c_uint32(0), ctypes.byref(temp)) == 0:
                    temp_c = int(temp.value)
            except Exception:
                pass
            try:
                if nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(power)) == 0:
                    power_w = round(power.value / 1000.0, 1)
            except Exception:
                pass
            rows.append({
                "index": str(i), "name": nm,
                "util": gpu_util or None, "usedMiB": used // (1024 * 1024),
                "totalMiB": total // (1024 * 1024),
                "tempC": temp_c or None, "powerW": power_w or None, "powerLimitW": None,
                "source": "nvml",
            })
    except Exception:
        return []
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass
    return rows


def sysinfo() -> dict:
    # 实时指标（利用率/温度/功耗）优先；万一 nvidia-smi / NVML 这一路拿不到，
    # 退回静态显卡列表（至少保证 name / totalMiB / freeMiB 有值），
    # 避免前端把"只是没读到实时值"误显示成"nvidia-smi 不可用"。
    gpus = gpu_metrics()
    if not gpus:
        gpus = gpu_list()
    return {
        "host": {
            "cpuName": cpu_name(),
            "cpuPercent": cpu_percent(),
            "cpuMHz": cpu_mhz(),
            "cores": os.cpu_count(),
            "ram": mem_info(),
        },
        "gpus": gpus,
        "ts": time.time(),
    }


# --------------------------------------------------------------------------- #
# llama-server 的运行态
# --------------------------------------------------------------------------- #

def port_listening(port) -> bool:
    """纯 TCP 探测：只要有人 listen 就算占用（比 HTTP 探测更可靠）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.35)
            return s.connect_ex(("127.0.0.1", int(port))) == 0
    except Exception:
        return False


def http_raw(port, path: str, timeout: float = 4.0) -> tuple[int, str] | None:
    """带状态码的 GET。llama-server 加载中会回 503，用 http_json 会把这个信息丢掉。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read().decode("utf-8", "replace")
        except Exception:
            return exc.code, ""
    except Exception:
        return None


def prom_metrics(port, timeout: float = 5.0) -> dict:
    """抓 llama-server 的 /metrics（Prometheus 文本），拍平成 {指标: 数值}。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        k, _, v = line.rpartition(" ")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


LLAMA_STATE_ZH = {
    "ok": "已就绪",
    "loading": "加载模型",
    "starting": "启动中",
    "stopped": "已停止",
    "down": "未监听",
    "error": "启动失败",
}


def llama_status(task: "Task | None", port=None) -> dict:
    port = str(port or (task.meta.get("port") if task else "") or "")
    state, label, percent, health = "down", "未监听", None, None
    listening = bool(port) and port_listening(port)

    if listening:
        probe = http_raw(port, "/health", 4.0)
        if probe:
            code, body = probe
            try:
                health = json.loads(body)
            except Exception:
                health = {"raw": body[:300]}
            st = str((health or {}).get("status") or "").lower()
            if code == 200 and st in ("ok", "ready"):
                state, label, percent = "ok", "已就绪", 100.0
            elif code == 503 or "load" in st:
                state, label = "loading", "加载模型"
                err = (health or {}).get("error")
                msg = err.get("message") if isinstance(err, dict) else (err if isinstance(err, str) else "")
                if msg and "load" not in str(msg).lower():
                    state, label = "error", str(msg)[:200]
            else:
                state, label = "error", str((health or {}).get("message") or f"HTTP {code}")[:200]

    running = bool(task and task.status == "running")
    if state == "down":
        if running:
            state, label = "starting", (task.phase or "启动中")
        elif task:
            state, label = "stopped", "已停止"

    # /props 和 /metrics 只在端口真的在监听时才去取 —— 否则每次轮询都要白等超时
    props = http_json(port, "/props", 4.0) if (listening and state in ("ok", "loading")) else None
    metrics = prom_metrics(port) if (listening and state == "ok") else {}
    gen = (props or {}).get("default_generation_settings") or {}

    return {
        "id": task.id if task else None,
        "port": port,
        "taskStatus": task.status if task else None,
        "anyRunning": running,
        "state": state,
        "label": label,
        "percent": percent,
        "health": health,
        "phase": task.phase if task else None,
        "pid": (task.proc.pid if task and task.proc else None),
        "elapsed": round((task.finished or time.time()) - task.started, 1) if task else None,
        "modelPath": (task.meta.get("modelPath") if task else None) or (props or {}).get("model_path"),
        "engine": task.meta.get("engine") if task else None,
        "props": {
            "modelAlias": (props or {}).get("model_alias"),
            "build": (props or {}).get("build_info"),
            "nCtx": gen.get("n_ctx"),
            "totalSlots": (props or {}).get("total_slots"),
            "modalities": (props or {}).get("modalities"),
            "isSleeping": (props or {}).get("is_sleeping"),
        } if props else None,
        "metrics": {
            "promptTokens": metrics.get("llamacpp:prompt_tokens_total"),
            "decodedTokens": metrics.get("llamacpp:tokens_predicted_total"),
            "decodeTps": metrics.get("llamacpp:predicted_tokens_seconds"),
            "prefillTps": metrics.get("llamacpp:prompt_tokens_seconds"),
            "processing": metrics.get("llamacpp:requests_processing"),
            "deferred": metrics.get("llamacpp:requests_deferred"),
        } if metrics else None,
    }


def build_llama_cmd(body: dict) -> tuple[list[str], dict]:
    """把请求体拼成 llama-server 命令行；返回 (cmd, meta)。"""
    exe = find_llama_exe(body.get("exe") or body.get("dir") or "")
    if not exe:
        raise ValueError("没找到 llama-server.exe —— 请指定 llama.cpp 安装目录（或直接指向该 exe）")

    model = str(body.get("model") or "").strip().strip('"')
    if not os.path.isfile(model):
        raise ValueError(f"模型文件不存在：{model}")

    caps = llama_caps(exe)
    port = str(body.get("port") or 8080).strip()

    def val(key, default=None):
        v = body.get(key)
        if v is None or str(v).strip() == "":
            return default
        return str(v).strip()

    n_cores = os.cpu_count() or 8
    ctx = val("ctx", "32768")
    ngl = val("ngl", "99")
    kv_k = val("kvK", "f16")
    kv_v = val("kvV", "q8_0")
    batch = val("batch", str(max(512, min(4096, n_cores * 256))))
    ubatch = val("ubatch", str(max(128, min(2048, n_cores * 128))))
    cache_ram = val("cacheRam", "8192")
    threads = val("threads", str(max(4, min(16, n_cores))))
    threads_batch = val("threadsBatch", str(max(4, min(24, n_cores + n_cores // 2))))
    parallel = val("parallel", "1")
    host = val("host", "127.0.0.1")
    alias = val("alias", os.path.splitext(os.path.basename(model))[0])

    # turbo KV 只有 turboquant 那套构建认；选错了就降级并如实告知
    notes: list[str] = []
    if (kv_k in TURBO_KV or kv_v in TURBO_KV) and not caps.get("turboKV"):
        notes.append(f"这个构建不支持 turbo KV（{kv_k}/{kv_v} 只存在于 llama-turboquant），"
                     f"已自动降级为 f16 / q8_0")
        kv_k, kv_v = "f16", "q8_0"

    cmd = [exe, "--model", model, "--host", host, "--port", port,
           "--alias", alias, "--ctx-size", ctx, "--n-gpu-layers", ngl,
           "--batch-size", batch, "--ubatch-size", ubatch,
           "--parallel", parallel,
           "--threads", threads, "--threads-batch", threads_batch,
           "--cache-type-k", kv_k, "--cache-type-v", kv_v]

    flash = val("flashAttn", "on")
    if flash and flash.lower() not in ("off", "0", "false"):
        cmd += ["--flash-attn", "on"]

    if caps.get("cacheRam") and cache_ram:
        cmd += ["--cache-ram", cache_ram]
    if caps.get("metrics"):
        cmd.append("--metrics")
    if caps.get("loadMode") and val("loadMode"):
        cmd += ["--load-mode", val("loadMode")]
    if caps.get("reasoning") and val("reasoning"):
        cmd += ["--reasoning", val("reasoning")]

    # logit-bias：用于按 token id 压制/提升特定词元。参考脚本用它关掉思考链
    # （--logit-bias 248066-inf,248067-inf），这里按逗号拆成多对直接透传。
    logit_bias = val("logitBias")
    if logit_bias:
        for pair in logit_bias.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if not re.match(r"^-?\d+[-,]\s*-?\d+", pair):
                continue
            cmd += ["--logit-bias", pair.replace(" ", "")]

    mmproj = val("mmproj")
    if mmproj:
        if not os.path.isfile(mmproj):
            raise ValueError(f"视觉投影文件不存在：{mmproj}")
        cmd += ["--mmproj", mmproj]

    # 默认不预热：省掉首启等待；档位里显式开了 warmup 才预热（参考脚本是开的）
    warmup = val("warmup")
    if warmup and warmup.lower() in ("1", "true", "on"):
        cmd.append("--warmup")
    else:
        cmd.append("--no-warmup")

    for extra in shlex_split(val("extraArgs") or ""):
        cmd.append(extra)

    return cmd, {
        "port": port, "modelPath": model, "engine": "llama",
        "exe": exe, "caps": caps, "kvK": kv_k, "kvV": kv_v,
        "ctx": ctx, "ngl": ngl, "notes": notes,
    }


def shlex_split(s: str) -> list[str]:
    """按空格切附加参数，但保留引号里的空格。"""
    out, cur, quote = [], "", ""
    for ch in s:
        if quote:
            if ch == quote:
                quote = ""
            else:
                cur += ch
        elif ch in "\"'":
            quote = ch
        elif ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# FreeToken 引擎：检测 + 一键安装 / 修复
# --------------------------------------------------------------------------- #
#
# 官方安装方式（https://github.com/FlashML-org/FreeToken）：
#   * 桌面版：flashml.ai 下载，装完自动配好引擎
#   * CLI   ：uv pip install "freetoken[accel]"
#   * 离线   ：跑 engine/install.ps1（旁边要有 dist\freetoken-*.whl + kernel-cache wheel）
#
# 本机就是第三种：%LOCALAPPDATA%\FreeToken 里是从本地 wheel 用 uv 装出来的。
# 所以"一键安装"按 官方脚本 → 本地 wheel + uv → 联网 uv 三级降级。

FT_DESKTOP_HINTS = [
    r"G:\AI\FreeToken Desktop", r"C:\AI\FreeToken Desktop", r"D:\AI\FreeToken Desktop",
    r"E:\AI\FreeToken Desktop",
]


def ft_home() -> str:
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "FreeToken")


def find_ft() -> list[str]:
    """找 ft.exe：先看安装记录 ft-bin.txt，再走常见路径，最后 PATH。"""
    cands: list[str] = []
    home = ft_home()
    if home:
        # 安装脚本会把真实路径写在 ft-bin.txt 里，最权威
        try:
            with open(os.path.join(home, "ft-bin.txt"), "r", encoding="utf-8",
                      errors="replace") as fh:
                p = fh.read().strip().strip('"')
            if p:
                cands.append(p)
        except OSError:
            pass
        cands.append(os.path.join(home, "venv", "Scripts", "ft.exe"))
        cands.append(os.path.join(home, "bin", "ft.exe"))
    cands += [
        r"C:\Program Files\FreeToken\venv\Scripts\ft.exe",
        os.path.join(HERE, "ft.exe"),
    ]
    found: list[str] = []
    for p in cands:
        if p and os.path.isfile(p) and p not in found:
            found.append(p)
    which = shutil.which("ft")
    if which and which not in found:
        found.insert(0, which)
    return found


def _find_uv() -> str | None:
    cands = [
        os.path.join(ft_home(), "uv", "uv.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".local", "bin", "uv.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "uv", "uv.exe"),
    ]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return shutil.which("uv")


def _ft_desktop_dirs() -> list[str]:
    """找到 FreeToken Desktop 的解压目录（里面应有 engine/install.ps1 + engine/dist）。"""
    out: list[str] = []
    seen = set()

    def push(d: str) -> None:
        d = os.path.abspath(d)
        k = d.lower()
        if k not in seen and os.path.isdir(d):
            seen.add(k)
            out.append(d)

    for h in FT_DESKTOP_HINTS:
        if os.path.isdir(h):
            push(h)
            push(os.path.join(h, "engine"))
    # 再浅扫一层常见根目录，找名字里带 freetoken 的文件夹
    for root in (r"G:\AI", r"C:\AI", r"D:\AI"):
        if not os.path.isdir(root):
            continue
        try:
            with os.scandir(root) as it:
                for e in it:
                    if not e.is_dir():
                        continue
                    if "freetoken" in e.name.lower():
                        push(e.path)
                        push(os.path.join(e.path, "engine"))
        except OSError:
            pass
    return out


def _find_ft_install_script() -> str | None:
    for d in _ft_desktop_dirs() + [ft_home()]:
        p = os.path.join(d, "install.ps1")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4000).lower()
        except OSError:
            continue
        if "freetoken" in head:
            return p
    return None


def _find_ft_wheels() -> tuple[str | None, str | None]:
    """返回 (runtime wheel, kernel-cache wheel)。"""
    runtime = kernel = None
    for d in _ft_desktop_dirs() + [ft_home()]:
        for sub in (d, os.path.join(d, "dist")):
            if not os.path.isdir(sub):
                continue
            try:
                with os.scandir(sub) as it:
                    for e in it:
                        if not e.is_file() or not e.name.lower().endswith(".whl"):
                            continue
                        low = e.name.lower()
                        if low.startswith("freetoken_kernel_cache") or low.startswith("freetoken-kernel-cache"):
                            kernel = kernel or e.path
                        elif low.startswith("freetoken-"):
                            runtime = runtime or e.path
            except OSError:
                continue
    return runtime, kernel


def _find_ft_desktop_exe() -> str | None:
    for d in _ft_desktop_dirs() + [ft_home()]:
        for name in ("freetoken-desktop.exe", "FreeToken Desktop.exe", "FreeToken.exe"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def ft_install_plan() -> dict:
    """按 官方脚本 → 本地 wheel → 联网 三级降级，挑一条能走通的安装路径。"""
    home = ft_home()
    venv = os.path.join(home, "venv")
    script = _find_ft_install_script()
    runtime, kernel = _find_ft_wheels()
    uv = _find_uv()

    if script:
        return {
            "mode": "script",
            "detail": f"使用官方安装脚本（离线，推荐）：{script}",
            "cmd": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", script],
            "cwd": os.path.dirname(script),
            "finds": {"script": script, "runtime": runtime, "kernel": kernel, "uv": uv},
        }

    if runtime and uv:
        parts = [f"& '{uv}' venv '{venv}' --python 3.12"]
        install = (f"& '{uv}' pip install --python '{venv}' --torch-backend=cu130 "
                   f"--reinstall-package freetoken --reinstall-package freetoken-kernel-cache "
                   f"'{runtime}'" + (f" '{kernel}'" if kernel else ""))
        ps = f"$ErrorActionPreference='Stop'; {parts[0]}; if ($?) {{ {install} }}"
        return {
            "mode": "wheel",
            "detail": f"用本地 uv + 本地 wheel 离线安装：{os.path.basename(runtime)}"
                      + (f" + {os.path.basename(kernel)}" if kernel else ""),
            "cmd": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            "cwd": os.path.dirname(runtime),
            "finds": {"runtime": runtime, "kernel": kernel, "uv": uv},
        }

    if uv:
        ps = (f"$ErrorActionPreference='Stop'; "
              f"if (-not (Test-Path '{venv}\\Scripts\\python.exe')) {{ & '{uv}' venv '{venv}' --python 3.12 }}; "
              f"& '{uv}' pip install --python '{venv}' --torch-backend=cu130 \"freetoken[accel]\"")
        return {
            "mode": "network",
            "detail": f"从 PyPI 在线安装 freetoken[accel]（用 {uv}）",
            "cmd": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            "finds": {"runtime": None, "kernel": None, "uv": uv},
        }

    # 连 uv 都没有 —— 先装 uv（官方渠道），再走联网安装
    ps = ("$ErrorActionPreference='Stop'; "
          "irm https://astral.sh/uv/install.ps1 | iex; "
          f"$uv = Join-Path $env:USERPROFILE '.local\\bin\\uv.exe'; "
          f"if (-not (Test-Path $uv)) {{ throw 'uv 安装失败，请手动安装 uv 后重试' }}; "
          f"& $uv venv '{venv}' --python 3.12; "
          f"& $uv pip install --python '{venv}' --torch-backend=cu130 \"freetoken[accel]\"")
    return {
        "mode": "bootstrap",
        "detail": "本机没有 uv，也没有离线 wheel —— 将先装 uv 再联网安装 freetoken[accel]",
        "cmd": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        "finds": {"runtime": None, "kernel": None, "uv": None},
    }


def ft_status() -> dict:
    home = ft_home()
    venv = os.path.join(home, "venv")
    found = find_ft()
    exe = found[0] if found else None
    installed = bool(exe and os.path.isfile(exe))

    version = ""
    if installed:
        out = _run_quiet([exe, "--version"], 40)
        m = re.search(r"freetoken\s+version\s+(\S+)", out) or re.search(r"version[: ]+(\S+)", out)
        version = m.group(1) if m else ""

    runtime, kernel = _find_ft_wheels()
    plan = ft_install_plan()
    return {
        "installed": installed,
        "exe": exe,
        "candidates": found,
        "version": version,
        "home": home,
        "venv": venv if os.path.isdir(venv) else None,
        "uv": _find_uv(),
        "installScript": _find_ft_install_script(),
        "desktopExe": _find_ft_desktop_exe(),
        "wheels": {"runtime": runtime, "kernel": kernel},
        "plan": {"mode": plan["mode"], "detail": plan["detail"]},
        "desktopUrl": "https://flashml.ai",
        "repoUrl": "https://github.com/FlashML-org/FreeToken",
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def weights_bytes(model: str) -> int:
    """模型目录里的权重总字节（.ftw / .safetensors / .gguf）。"""
    total = 0
    try:
        for entry in os.scandir(model):
            if entry.is_file() and entry.name.lower().endswith(WEIGHT_EXT):
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


GIB = 1024 ** 3


def _load_text_config(model_path: str) -> dict:
    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return {}
    try:
        with open(cfg, "r", encoding="utf-8") as fh:
            j = json.load(fh)
    except Exception:
        return {}
    return j.get("text_config") or j


def gdn_state_per_req(text_cfg: dict, dtype_bytes: int = 2) -> tuple[int, int]:
    """混合线性注意力（GDN）模型：单个请求的线性状态字节数，以及线性层数。

    与引擎 ``linear_state_bytes_per_req`` 同一套算法：
        conv 状态  = conv_dim x (kernel - 1) x 模型 dtype
        递推状态   = v_heads x key_head_dim x value_head_dim x fp32
    非混合模型（没有 linear_attention 层）返回 0。
    """
    layers = text_cfg.get("layer_types") or []
    n_linear = sum(1 for t in layers if t == "linear_attention")
    if not n_linear:
        return 0, 0
    nk = text_cfg.get("linear_num_key_heads", 0)
    kd = text_cfg.get("linear_key_head_dim", 0)
    nv = text_cfg.get("linear_num_value_heads", 0)
    vd = text_cfg.get("linear_value_head_dim", 0)
    ker = text_cfg.get("linear_conv_kernel_dim", 4)
    conv_dim = nk * kd + nv * vd
    conv_bytes = conv_dim * max(0, ker - 1) * dtype_bytes
    rec_bytes = nv * kd * vd * 4            # mamba_ssm_dtype 默认 fp32
    return n_linear * (conv_bytes + rec_bytes), n_linear


def state_slots(max_running_req: int, cache_type: str, cache_ratio: float = 2.0) -> int:
    """引擎 ``_linear_pool_num_slots``：GDN 状态池的物理槽位数。

    这是这台机器上最容易被忽略的显存大户——radix 下是 ``4×并发 + max(4, 2×并发) + 1``，
    并发 1 只要 9 槽，并发 4 直接跳到 25 槽。naive 只保留 ``并发 + 1``。
    """
    mr = max(1, int(max_running_req))
    if cache_type != "radix":
        return mr + 1
    n_cache = max(4, int(cache_ratio * mr))
    return 4 * mr + n_cache + 1


def preflight(model_path: str, gpu_index: int = 0, want_conc: int = 4,
               cache_type: str | None = None) -> dict:
    """按显存预算公式给这台机器上这个模型反推可行的启动参数。

        可用池 = memory_ratio x 加载前空闲显存 - 权重 - 固定缓存（这里是 GDN 状态池）
        KV     = 可用池 - GDN 状态池

    KV 一旦算成 ≤ 0，引擎就会抛 "Not enough memory for KV cache"。

    ``cache_type`` 是用户的硬约束（如果给了）：preflight 必须按用户指定的 cacheType
    算 GDN 状态池字节数并判断是否够；不够时直接 409 拒启，不再静默改用别的 cacheType。
    """
    w = weights_bytes(model_path)
    text_cfg = _load_text_config(model_path)
    per_req, n_linear = gdn_state_per_req(text_cfg)

    gpus = gpu_list()
    gpu = next((g for g in gpus if str(g.get("index")) == str(gpu_index)), None) \
        or (gpus[0] if gpus else None)
    free = int(gpu["freeMiB"]) * 1024 * 1024 if gpu and gpu.get("freeMiB") else 0

    kv_floor = int(1.0 * GIB)

    # 显存预算里的"余量"要给 CUDA graph 捕获 + 激活值 + 权重上传缓冲留空间，
    # 否则 KV 算得下、权重也装得进，但一捕获 CUDA graph 就 OOM。
    # 但余量卡太死会把"差一点点"的模型直接判死（例如 18.8G 权重 / 22.7G 空闲），
    # 所以分两档：
    #   safe        —— 留 3.5 GiB 或 15%，ratio ≤ 0.90，最稳
    #   aggressive  —— 只留 1.5 GiB 或 6%，ratio ≤ 0.95，并把 CUDA graph batch 压到 1
    # 先试安全档；全不满足再退极限档，让用户至少有机会跑起来（前端会标风险）。
    T_CAP_SAFE, T_CAP_AGG = 0.90, 0.95
    tiers = []
    if free:
        tiers.append(("safe", int(max(3.5 * GIB, 0.15 * free)), T_CAP_SAFE))
        tiers.append(("aggressive", int(max(1.5 * GIB, 0.06 * free)), T_CAP_AGG))
    else:
        tiers.append(("safe", 0, T_CAP_SAFE))

    def _mk_options(pool_v):
        opts = []
        cache_types = (cache_type,) if cache_type else ("radix", "naive")
        for m in (8, 4, 2, 1):
            for ct in cache_types:
                slots_v = state_slots(m, ct)
                state_v = per_req * slots_v
                kv = pool_v - state_v
                opts.append({
                    "maxRunningReq": m,
                    "cacheType": ct,
                    "slots": slots_v,
                    "stateGB": round(state_v / GIB, 2),
                    "kvGB": round(kv / GIB, 2),
                    "ok": bool(free) and kv >= kv_floor,
                    "note": "跨请求前缀复用（更快）" if ct == "radix" else "更省状态池，放弃跨请求前缀复用",
                })
        opts.sort(key=lambda o: (not o["ok"], o["cacheType"] != "radix", -o["maxRunningReq"]))
        return opts


    def _pick(usable):
        """满足目标并发的前提下取并发最小的（给 KV 留更多余量），同并发优先 radix。

        用户指定 cache_type 时，"并发不足"是用户自身的硬约束：直接返回 None 让上层
        409 拒启，不要静默改成别的 cacheType 骗用户"能跑"。
        """
        if not usable:
            return None
        fit = [o for o in usable if o["maxRunningReq"] >= want_conc]
        if fit:
            fit.sort(key=lambda o: (o["maxRunningReq"], o["cacheType"] != "radix"))
            return fit[0]
        if cache_type:
            # 用户硬约束了 cacheType，但该类型下没有任何满足 want_conc 的方案；
            # 不再降并发（用户没说要并发 1）或换 cacheType，直接报"装不下"。
            return None
        radix_only = [o for o in usable if o["cacheType"] == "radix"]
        return max(radix_only or usable, key=lambda o: (o["maxRunningReq"], o["kvGB"]))

    tier = "safe"
    ratio = round(min(T_CAP_SAFE, max(0.72, (free - tiers[0][1]) / free)), 3) if free else T_CAP_SAFE
    pool = int(ratio * free) - w if free else 0
    headroom = free - int(ratio * free) if free else 0
    options = _mk_options(pool)
    best = None
    for name, reserve, cap in tiers:
        r_t = round(min(cap, max(0.72, (free - reserve) / free)), 3) if free else cap
        pool_t = int(r_t * free) - w if free else 0
        opts_t = _mk_options(pool_t)
        best_t = _pick([o for o in opts_t if o["ok"]])
        if best_t:
            tier, ratio, pool = name, r_t, pool_t
            headroom = free - int(r_t * free) if free else 0
            options, best = opts_t, best_t
            break
        if name == "aggressive":
            # 两档都不行：保留极限档的数字给用户看（更接近实际），best 仍为 None
            tier, ratio, pool = name, r_t, pool_t
            headroom = free - int(r_t * free) if free else 0
            options = opts_t

    if best and free - w < 3 * GIB:
        prefill = 2048
    elif best and free - w < 4 * GIB:
        prefill = 4096
    else:
        prefill = 4096

    warns = []
    if not w:
        warns.append("这个目录里没找到权重文件（.ftw / .safetensors / .gguf），请确认路径。")
    if not os.path.isfile(os.path.join(model_path, "freetoken_weight.json")):
        warns.append("这不是 FTW 目录，按原始 HF 加载会明显更慢、更吃显存。")
    if not free:
        # 显卡探测没拿到数据。不单独说清楚的话，下面会把「读不到显存」误报成「显存不够」。
        warns.append(
            "⚠ 读不到显卡空闲显存（nvidia-smi / NVML 都没返回数据），显存预算没法算准。"
            "最常见原因是工具被 **32 位 Python** 启动了：32 位进程看不到 System32 里的 "
            "nvidia-smi（已自动改走 Sysnative 通道）；也可能是驱动异常或显卡被独占。"
            "建议用 64 位 Python 重开（启动工具.bat 会自动优先 64 位）。")
    if tier == "aggressive" and best:
        warns.append(
            f"⚠ 安全余量装不下，已切到「极限档」：memory-ratio={ratio}、CUDA graph batch 压到 1、"
            f"只留 {headroom / GIB:.1f} GiB 给激活值。能跑，但长上下文或高并发时仍可能 OOM —— "
            f"更稳的做法是先关掉别的占显存程序，把空闲显存抬到 "
            f"{(w + per_req * state_slots(1, 'radix') + 1.5 * GIB) / GIB:.1f} GiB 以上。")
    if not best and cache_type:
        # 用户硬约束了 cacheType，但该类型下没有任何满足 want_conc 的方案。
        # 给出针对该 cacheType 的诊断：naive 在这个显存下能装，radix 不行，等等。
        need_radix = (w + per_req * state_slots(want_conc, "radix") + kv_floor) / GIB
        need_naive = (w + per_req * state_slots(want_conc, "naive") + kv_floor) / GIB
        warns.append(
            f"你选了 cache-type={cache_type}，但当前空闲显存 {free / GIB:.1f} GiB 装不下"
            f"「权重 {w / GIB:.1f} GiB + GDN 状态池 + KV」并发 {want_conc} 的配置。"
            f"radix 下并发 {want_conc} 需要 {state_slots(want_conc, 'radix')} 个 GDN 槽位"
            f"（约 {per_req * state_slots(want_conc, 'radix') / GIB:.2f} GiB），"
            f"naive 下只要 {state_slots(want_conc, 'naive')} 个（约 {per_req * state_slots(want_conc, 'naive') / GIB:.2f} GiB）。"
            f"想要并发 {want_conc} 至少需要 {need_radix:.1f} GiB（radix）/{need_naive:.1f} GiB（naive）。"
            "建议：① 把 cache-type 改成 naive（最常见方案）；② 或降低并发；③ 或换更小的量化版本。")
    elif not best:
        need = (w + per_req * state_slots(1, "naive") + kv_floor) / GIB
        warns.append(
            f"当前空闲显存 {free / GIB:.1f} GiB 装不下「权重 {w / GIB:.1f} GiB + GDN 状态池 + KV」。"
            f"就算用最省的并发 1 + naive，也至少要 {need:.1f} GiB 空闲显存。"
            "先关掉其它占显存的程序（或换更小的量化版本）再启动。")
        warns.append(
            f"目标并发 {want_conc} 在 radix 下需要 {state_slots(want_conc, 'radix')} 个 GDN 槽位"
            f"（约 {per_req * state_slots(want_conc, 'radix') / GIB:.2f} GiB），装不下；"
            f"已自动降到并发 {best['maxRunningReq']}。想强行上高并发，可改用 naive 缓存。")
    if n_linear:
        warns.append(
            f"该模型有 {n_linear} 层线性注意力，状态池按并发倍数增长"
            f"（单请求约 {per_req / GIB:.2f} GiB），这是显存预算里最容易忽略的一项。")

    return {
        "modelPath": model_path,
        "weightsBytes": w, "weightsGB": round(w / GIB, 2),
        "freeBytes": free, "freeGB": round(free / GIB, 2),
        "gpu": gpu,
        "ratio": ratio,
        "tier": tier,                      # safe | aggressive
        "poolGB": round(pool / GIB, 2),
        "headroomGB": round(headroom / GIB, 2),
        "gdnPerReqGB": round(per_req / GIB, 3),
        "gdnLayers": n_linear,
        "options": options,
        "recommended": {
            "memoryRatio": ratio,
            "maxRunningReq": best["maxRunningReq"] if best else 1,
            "cacheType": best["cacheType"] if best else "naive",
            "maxPrefillLength": prefill,
            # 极限档把 CUDA graph 的 batch 压到 1，能省下相当一块捕获显存
            "cudaGraphMaxBs": "1" if tier == "aggressive" else "",
        } if best else None,
        "warnings": warns,
        "isFTW": os.path.isfile(os.path.join(model_path, "freetoken_weight.json")),
    }


def http_json(port, path: str, timeout: float = 4.0) -> dict | None:
    """向模型服务发一个只读 GET；连不上返回 None。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


# 端口发现：工具重启过、或者服务是 FreeToken Desktop / 命令行起的，都靠这个找回来
DEFAULT_SCAN_PORTS = list(range(8421, 8431)) + [1919, 8000, 8080, 8081]


def discover_services(ports: list[int] | None = None, timeout: float = 0.6) -> list[dict]:
    ports = ports or DEFAULT_SCAN_PORTS
    found: list[dict] = []
    lock = threading.Lock()

    def probe(p: int) -> None:
        h = http_json(p, "/health", timeout)
        if not isinstance(h, dict):
            return
        with lock:
            found.append({"port": str(p), "health": h,
                          "state": h.get("status") or "?",
                          "model": h.get("model")})

    threads = [threading.Thread(target=probe, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 0.4)
    found.sort(key=lambda x: int(x["port"]))
    return found


# 引擎 /health 的 loading 阶段名 -> 中文
LOAD_PHASE_ZH = {
    "weights": "加载权重",
    "model": "加载模型",
    "graph": "捕获 CUDA graph",
    "cuda_graph": "捕获 CUDA graph",
    "warmup": "预热",
    "tokenizer": "加载分词器",
    "other": "初始化",
}


def serve_status(task: "Task | None", port: str | None = None) -> dict:
    """拼出模型服务的实时状态。

    引擎自己在 ``/health`` 里就给了权威答案：``loading``（还带阶段和字节进度）
    / ``ok`` / ``error``。比 ``/v1/models`` 靠谱得多 —— 后者在权重还没加载完时
    就已经返回 200 了，用它判断"就绪"会骗人。
    """
    if task is not None:
        port = port or task.meta.get("port")
    port = str(port or "")

    health = http_json(port, "/health", 4.0) if port else None

    if health is None:
        state, label, percent = "down", "未监听", None
    elif health.get("status") == "loading":
        state = "loading"
        raw = health.get("phase") or "other"
        label = LOAD_PHASE_ZH.get(raw, raw)
        pr = health.get("progress") or {}
        done, total = pr.get("done_bytes") or 0, pr.get("total_bytes") or 0
        percent = round(done * 100.0 / total, 1) if total else None
    elif health.get("status") == "ok":
        state, label, percent = "ok", "已就绪", 100.0
    else:
        state, label, percent = "error", health.get("message") or "启动失败", None

    task_running = bool(task and task.status == "running")
    if state == "down" and task and task.status != "running":
        state = "stopped"
        label = "已停止"

    return {
        "id": task.id if task else None,
        "port": port,
        "taskStatus": task.status if task else None,
        "anyRunning": task_running,
        "state": state,                    # down | loading | ok | error | stopped
        "label": label,
        "percent": percent,
        "health": health,
        "phase": task.phase if task else None,
        "pid": (task.proc.pid if task and task.proc else None),
        "elapsed": round((task.finished or time.time()) - task.started, 1) if task else None,
        "modelPath": task.meta.get("modelPath") if task else None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "FTWTool/1.0"
    protocol_version = "HTTP/1.1"

    # -- 基础工具 ---------------------------------------------------------- #
    def log_message(self, fmt, *args):        # 静音
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, msg: str, code: int = 400) -> None:
        self._json({"error": str(msg)}, code)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # -- GET --------------------------------------------------------------- #
    def do_GET(self):
        path, _, query = self.path.partition("?")
        qs = {}
        for kv in query.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                qs[k] = urllib.parse.unquote_plus(v)

        if path in ("/", "/index.html"):
            try:
                with open(INDEX, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, b"index.html not found", "text/plain")

        if path == "/api/env":
            fts = find_ft()
            gpus = gpu_list()
            return self._json({
                "ft": {"paths": fts, "active": fts[0] if fts else None, "ok": bool(fts)},
                "gpus": gpus,
                "drives": drive_list(),
                "platform": sys.platform,
                "python": sys.version.split()[0],
                # --- 显卡探测诊断：出问题时能一眼看出卡在哪一环 ---
                "pythonBits": _host_bits(),          # 32 位解释器是显卡探测失败的头号原因
                "pythonExe": sys.executable,
                "wow64": _is_wow64(),
                "smiPath": _nvidia_smi_path(),       # 实际选中的 nvidia-smi（null = 一个都没跑通）
                "smiSource": (gpus[0].get("source") if gpus else None),
                "cwd": HERE,
            })

        if path == "/api/fs/list":
            exts = tuple(x.strip().lower() for x in (qs.get("ext") or "").split(",") if x.strip())
            with_files = str(qs.get("files") or "").lower() in ("1", "true", "yes")
            return self._json(self._list_dir(qs.get("path") or "", with_files, exts))

        if path == "/api/fs/scan":
            root = qs.get("root") or ""
            depth = int(qs.get("depth") or 2)
            if not os.path.isdir(root):
                return self._err(f"目录不存在: {root}")
            mode = (qs.get("mode") or "hf").lower()
            try:
                if mode == "gguf":
                    # 模型库的"GGUF 文件"模式：列出目录里的独立 .gguf（不要求 HF 结构）
                    models = scan_gguf(root, depth)
                    return self._json({"root": root, "mode": "gguf", "models": models})
                models = scan_models(root, depth)
                return self._json({"root": root, "mode": "hf", "models": models})
            except Exception as exc:
                return self._err(exc)

        if path == "/api/fs/inspect":
            p = qs.get("path") or ""
            if not os.path.isdir(p):
                return self._err(f"目录不存在: {p}")
            return self._json({"path": p, **is_model_dir(p)})

        if path == "/api/preflight":
            p = qs.get("path") or ""
            if not os.path.isdir(p):
                return self._err(f"目录不存在: {p}")
            try:
                return self._json(preflight(p, int(qs.get("gpu") or 0),
                                            int(qs.get("conc") or 4)))
            except Exception as exc:
                return self._err(exc)

        if path == "/api/task":
            task = TASKS.get(qs.get("id", ""))
            if not task:
                return self._err("任务不存在", 404)
            return self._json(task.snapshot(int(qs.get("offset") or 0)))

        if path == "/api/tasks":
            with TASKS_LOCK:
                items = [t.snapshot(0) for t in TASKS.values()]
            items.sort(key=lambda t: t["elapsed"])
            return self._json({"tasks": items})

        if path == "/api/serve/status":
            task = TASKS.get(qs.get("id", ""))
            if not task:
                # 没给任务 id 时（例如页面刷新后），退化成按端口直接探测
                port = qs.get("port") or ""
                if not port:
                    return self._err("任务不存在", 404)
                return self._json(serve_status(None, port))
            return self._json(serve_status(task))

        if path == "/api/serve/stats":
            port = qs.get("port") or "8421"
            return self._json({"stats": http_json(port, "/v1/stats", 6.0)})

        if path == "/api/gpu":
            return self._json({"gpus": gpu_list()})

        if path == "/api/serve/discover":
            ports = None
            if qs.get("ports"):
                ports = [int(x) for x in qs["ports"].split(",") if x.strip().isdigit()]
            try:
                return self._json({"services": discover_services(ports)})
            except Exception as exc:
                return self._err(exc)

        # ---- llama.cpp / llama-turboquant ---- #
        if path == "/api/sysinfo":
            return self._json(sysinfo())

        if path == "/api/llama/env":
            roots = [x for x in (qs.get("roots") or "").split(os.pathsep) if x.strip()]
            installs = detect_llama_installs(roots)
            return self._json({
                "installs": installs,
                "presets": LLAMA_PRESETS,
                "turboKVTypes": list(TURBO_KV),
                "defaultGgufRoot": roots[0] if roots else r"G:\AI",
                "sources": {
                    "llama.cpp": "https://github.com/ggml-org/llama.cpp",
                    "turboquant": "https://github.com/TheTom/llama-cpp-turboquant",
                    "freetoken": "https://github.com/FlashML-org/FreeToken/tree/main",
                },
            })

        if path == "/api/llama/scan":
            root = (qs.get("dir") or "").strip()
            if not root or not os.path.isdir(root):
                return self._err(f"目录不存在: {root or '(空)'}")
            try:
                models = scan_gguf(root, int(qs.get("depth") or 2))
            except Exception as exc:
                return self._err(exc)
            return self._json({"dir": root, "models": models,
                               "count": len(models),
                               "mmproj": [m for m in models if m["mmproj"]]})

        if path == "/api/llama/status":
            task = TASKS.get(qs.get("id", "")) or None
            port = qs.get("port") or ""
            if not task and not port:
                return self._err("任务不存在", 404)
            return self._json(llama_status(task, port or None))

        # 取最近一次 llama.cpp 启动任务的完整日志
        if path == "/api/llama/last-log":
            engine = qs.get("engine") or ""
            with TASKS_LOCK:
                tasks = [t for t in TASKS.values() if t.kind == "llama"]
                if engine:
                    tasks = [t for t in tasks if (t.meta or {}).get("engine") == engine]
            tasks.sort(key=lambda t: t.started, reverse=True)
            if not tasks:
                return self._json({"task": None, "lines": []})
            t = tasks[0]
            with t.lock:
                lines = list(t.lines)
            return self._json({
                "task": {
                    "id": t.id, "status": t.status, "phase": t.phase,
                    "exitCode": t.exit_code, "elapsed": (t.finished or time.time()) - t.started,
                    "cmd": t.cmd, "meta": t.meta, "progress": t.progress,
                },
                "lines": lines,
            })

        if path == "/api/llama/pick-exe":
            p = qs.get("path") or ""
            exe = find_llama_exe(p)
            if not exe:
                return self._err("这个路径下没有 llama-server.exe")
            return self._json({"exe": exe, "dir": os.path.dirname(exe),
                               "caps": llama_caps(exe)})

        # ---- FreeToken 引擎 ---- #
        if path == "/api/ft/status":
            return self._json(ft_status())

        # 取最近一次 FreeToken（serve）任务的完整日志，引擎没启动也能拿，
        # 方便排查 "按了启动但什么都没出现" 的情况。
        if path == "/api/serve/last-log":
            kind = qs.get("kind") or "serve"
            with TASKS_LOCK:
                tasks = [t for t in TASKS.values() if t.kind == kind]
            tasks.sort(key=lambda t: t.started, reverse=True)
            if not tasks:
                return self._json({"task": None, "lines": []})
            t = tasks[0]
            with t.lock:
                lines = list(t.lines)
            return self._json({
                "task": {
                    "id": t.id, "status": t.status, "phase": t.phase,
                    "exitCode": t.exit_code, "elapsed": (t.finished or time.time()) - t.started,
                    "cmd": t.cmd, "meta": t.meta, "progress": t.progress,
                },
                "lines": lines,
            })

        return self._err("未知接口", 404)

    # -- POST -------------------------------------------------------------- #
    def do_POST(self):
        path = self.path.partition("?")[0]
        body = self._body()

        if path == "/api/convert/start":
            return self._convert_start(body)
        if path == "/api/serve/start":
            return self._serve_start(body)
        if path == "/api/task/cancel":
            task = TASKS.get(body.get("id", ""))
            if not task or not task.proc:
                return self._err("任务不存在或已结束", 404)
            task.status = "cancelled"
            task.finished = time.time()
            _kill_tree(task.proc.pid)
            task.add("[已取消] 进程树已终止")
            return self._json({"ok": True})
        if path == "/api/fs/reveal":
            p = body.get("path") or ""
            if os.path.isdir(p):
                try:
                    if os.name == "nt":
                        os.startfile(p)          # noqa: S606
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", p])
                    else:
                        subprocess.Popen(["xdg-open", p])
                    return self._json({"ok": True})
                except Exception as exc:
                    return self._err(exc)
            return self._err("目录不存在")
        if path == "/api/chat":
            return self._chat(body)
        if path == "/api/serve/stop":
            task = TASKS.get(body.get("id", ""))
            if not task:
                return self._err("任务不存在", 404)
            if task.proc and task.status == "running":
                _kill_tree(task.proc.pid)
                task.status = "cancelled"
                task.finished = time.time()
            return self._json({"ok": True})

        # ---- llama.cpp / llama-turboquant ---- #
        if path == "/api/llama/start":
            return self._llama_start(body)
        if path == "/api/llama/stop":
            task = TASKS.get(body.get("id", ""))
            if not task:
                return self._err("任务不存在", 404)
            if task.proc and task.status == "running":
                _kill_tree(task.proc.pid)
                task.status = "cancelled"
                task.finished = time.time()
                task.add("[已停止] 进程树已终止，显存归还中")
                task.phase = "已停止"
            return self._json({"ok": True})

        # ---- FreeToken 一键安装 / 修复 ---- #
        if path == "/api/ft/install":
            plan = ft_install_plan()
            if not plan.get("cmd"):
                return self._err(plan.get("detail") or "找不到可用的安装方式")
            task = _start_task("ftinstall", plan["cmd"], {
                "mode": plan["mode"], "detail": plan["detail"], "cwd": plan.get("cwd"),
            })
            return self._json({"taskId": task.id, "mode": plan["mode"],
                               "detail": plan["detail"], "cmd": plan["cmd"]})

        # ---- 退出工具：先收掉所有托管服务，再停 HTTP 服务 ---- #
        if path == "/api/shutdown":
            self._json({"ok": True})
            def _do_exit():
                _shutdown_children()
                try:
                    self.server.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_do_exit, daemon=True).start()
            return

        return self._err("未知接口", 404)

    # -- 目录浏览 ---------------------------------------------------------- #
    def _list_dir(self, path: str, with_files: bool = False,
                  exts: tuple = (), limit: int = 800) -> dict:
        """列目录。with_files=True 时同时列出指定后缀的文件（模型 / exe 选择器要用）。"""
        if not path:
            path = os.path.abspath(os.sep)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return {"error": f"目录不存在: {path}", "path": path,
                    "dirs": [], "files": [], "parent": None, "drives": drive_list()}
        dirs, files = [], []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        if e.is_dir():
                            if e.name.startswith("$"):
                                continue
                            st = e.stat()
                            dirs.append({
                                "name": e.name, "path": e.path,
                                "mtime": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                                "isModel": is_model_dir(e.path) if self._looks_like_model(e.path) else False,
                            })
                        elif with_files and e.is_file():
                            low = e.name.lower()
                            if exts and not any(low.endswith(x) for x in exts):
                                continue
                            st = e.stat()
                            files.append({
                                "name": e.name, "path": e.path,
                                "sizeBytes": st.st_size,
                                "sizeGB": round(st.st_size / GIB, 2),
                                "mtime": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                            })
                    except OSError:
                        continue
        except PermissionError:
            return {"error": "没有权限读取该目录", "path": path, "dirs": [], "files": [],
                    "parent": os.path.dirname(path), "drives": drive_list()}
        dirs.sort(key=lambda d: d["name"].lower())
        files.sort(key=lambda f: -f["sizeBytes"])
        if len(files) > limit:
            files = files[:limit]
        parent = os.path.dirname(path.rstrip("\\/")) or None
        return {"path": path, "parent": parent, "dirs": dirs, "files": files,
                "drives": drive_list()}

    @staticmethod
    def _looks_like_model(path: str, limit: int = 200) -> bool:
        """轻量探测：只看文件名，不做统计，避免浏览时卡顿。"""
        try:
            with os.scandir(path) as it:
                for i, e in enumerate(it):
                    if i > limit:
                        break
                    n = e.name.lower()
                    if n.endswith((".safetensors", ".gguf")) or n == "freetoken_weight.json":
                        return True
        except OSError:
            pass
        return False

    # -- 转换 -------------------------------------------------------------- #
    def _convert_start(self, body: dict):
        ft = (body.get("ft") or "").strip()
        if not ft or not os.path.isfile(ft):
            found = find_ft()
            if not found:
                return self._err("找不到 ft.exe，请在「设置」里手动指定路径")
            ft = found[0]

        model = (body.get("modelPath") or "").strip()
        if not os.path.isdir(model):
            return self._err(f"源目录不存在: {model}")
        out = (body.get("outPath") or "").strip()
        if not out:
            out = os.path.join(os.path.dirname(model),
                               os.path.basename(model.rstrip("\\/")) + "-FTW")

        if os.path.abspath(out) == os.path.abspath(model):
            return self._err("输出目录不能和源目录相同")

        src = is_model_dir(model)
        if not (src["safetensors"] or src["gguf"]):
            return self._err("源目录里没找到 .safetensors / .gguf 权重文件")

        # 空间预估：参考官方脚本，约等于源权重的 0.92 倍
        need = int(src["bytes"] * 0.92) + 500 * 1024 * 1024
        drive = os.path.splitdrive(os.path.abspath(out))[0] + "\\"
        try:
            free = shutil.disk_usage(drive).free
        except OSError:
            free = None
        if free is not None and free < need:
            return self._err(
                f"{drive} 空间不足：预计需要 {need / 2**30:.1f} GB，当前剩余 {free / 2**30:.1f} GB")

        if os.path.isdir(out) and os.listdir(out):
            if not body.get("overwrite"):
                return self._json({
                    "needConfirm": True,
                    "reason": "输出目录已存在且非空",
                    "outPath": out,
                    "estimatedGB": round(need / 2**30, 1),
                    "freeGB": round(free / 2**30, 1) if free else None,
                    "message": f"目录 {out} 已存在且非空。继续将**先删除该目录的全部内容**，然后重新转换。",
                })
            shutil.rmtree(out, ignore_errors=True)

        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

        cmd = [ft, "checkpoint", "--model", model, "--out", out,
               "--dtype", body.get("dtype") or "bfloat16",
               "--moe-backend", body.get("moeBackend") or "offload",
               "--shard-gib", str(body.get("shardGiB") or 8)]
        gpu = (body.get("gpu") or "").strip()
        if gpu:
            cmd += ["--gpu", gpu]
        for extra in (body.get("extraArgs") or "").split():
            cmd.append(extra)

        task = _start_task("convert", cmd, {
            "modelPath": model, "outPath": out,
            "estimatedGB": round(need / 2**30, 1),
            "sourceGB": round(src["bytes"] / 2**30, 1),
            "fileCount": src["safetensors"] + src["gguf"],
        })
        return self._json({"taskId": task.id, "cmd": cmd, "outPath": out,
                           "estimatedGB": round(need / 2**30, 1)})

    # -- 服务 -------------------------------------------------------------- #
    def _serve_start(self, body: dict):
        ft = (body.get("ft") or "").strip()
        if not ft or not os.path.isfile(ft):
            found = find_ft()
            if not found:
                return self._err("找不到 ft.exe")
            ft = found[0]
        model = (body.get("modelPath") or "").strip()
        if not os.path.isdir(model):
            return self._err(f"模型目录不存在: {model}")
        port = str(body.get("port") or 8421)

        # 同一个端口上已经有服务在跑：如果是本工具托管的上一个实例，先停掉再起；
        # 否则（别人占着）才报冲突，避免"先手动停"的麻烦。
        if self._probe(port):
            ok, err = _stop_existing_on_port(port, "FreeToken")
            if not ok:
                return self._err(err, 409)
            task.add("[端口曾被占用] 已接管并停止旧的服务实例，准备重新启动")

        # ---- 显存预算 ----
        # 总是按这台机器的实际情况跑一次 preflight，按结果**强制**填上 cacheType /
        # memory-ratio / cuda-graph-max-bs 等关键参数。用户填的部分字段保留（不被覆盖），
        # 这样无论用户是点了"启动"没调优、改了 cacheType 还是手动填了 memory-ratio，
        # 都跑不出能立刻 OOM 的配置。
        plan = None
        try:
            plan = preflight(
                model,
                int(body.get("gpu") or 0),
                int(body.get("maxRunningReq") or 4),
                # 把用户已经填上的字段作为「约束」传给 preflight：
                # 让它按用户的 cacheType 真正算状态池字节数。
                cache_type=str(body.get("cacheType") or "").strip() or None,
            )
        except Exception:
            plan = None
        if plan:
            rec = plan.get("recommended")
            if not rec:
                # preflight 找不到任何一个能跑的配置：直接 409 拒绝，不要拉起 ft.exe 再 OOM。
                msg = "按当前显存算不出能跑起来的 FreeToken 配置。"
                ws = plan.get("warnings") or []
                if ws:
                    msg += " " + "　".join(ws[:2])
                return self._json(
                    {"error": msg, "taskId": None, "ok": False, "reason": "preflight",
                     "message": msg, "plan": plan, "warnings": ws},
                    409,
                )
            # 用户没显式填过的字段，按推荐填上
            if not str(body.get("memoryRatio") or "").strip():
                body["memoryRatio"] = rec["memoryRatio"]
            if not str(body.get("cacheType") or "").strip():
                body["cacheType"] = rec["cacheType"]
            if not str(body.get("maxPrefillLength") or "").strip():
                body["maxPrefillLength"] = rec["maxPrefillLength"]
            if rec.get("cudaGraphMaxBs") and not str(body.get("cudaGraphMaxBs") or "").strip():
                body["cudaGraphMaxBs"] = rec["cudaGraphMaxBs"]

        cmd = [ft, "serve", "--model-path", model, "--port", port]
        optional = [
            ("maxRunningReq", "--max-running-requests"),
            ("memoryRatio", "--memory-ratio"),
            ("maxPrefillLength", "--max-prefill-length"),
            ("maxSeqLen", "--max-seq-len-override"),
            ("cacheType", "--cache-type"),
            ("attentionBackend", "--attention-backend"),
            ("nvfp4Backend", "--nvfp4-backend"),
            ("cudaGraphMaxBs", "--cuda-graph-max-bs"),
            ("servedModelName", "--served-model-name"),
            ("tensorParallelSize", "--tensor-parallel-size"),
        ]
        for key, flag in optional:
            val = str(body.get(key) or "").strip()
            if val:
                cmd += [flag, val]
        gpu = (body.get("gpu") or "").strip()
        if gpu:
            cmd += ["--gpu", gpu]
        for extra in (body.get("extraArgs") or "").split():
            cmd.append(extra)

        task = _start_task("serve", cmd, {"port": port, "modelPath": model})
        return self._json({"taskId": task.id, "cmd": cmd, "port": port, "plan": plan})

    # -- llama.cpp --------------------------------------------------------- #
    def _llama_start(self, body: dict):
        try:
            cmd, meta = build_llama_cmd(body)
        except ValueError as exc:
            return self._err(str(exc))

        port = meta["port"]
        # 端口被占：本工具托管的旧实例先接管停掉，否则报冲突
        if port_listening(port):
            ok, err = _stop_existing_on_port(port, "llama")
            if not ok:
                return self._err(err, 409)
            warns.append("[端口曾被占用] 已接管并停止旧的 llama-server，准备重新启动")

        # 用 mmap 时 llama.cpp 不再按 GPU 层数报错，但如果显存不够会在日志里炸，
        # 这里只做一次温和的容量提示，不阻断启动。
        warns = list(meta.get("notes") or [])
        try:
            gpus = gpu_metrics()
            g = gpus[0] if gpus else None
            model_gb = os.path.getsize(meta["modelPath"]) / GIB
            if g and g.get("totalMiB"):
                free_gb = (g["totalMiB"] - g["usedMiB"]) / 1024
                if model_gb > free_gb + 1.0:
                    warns.append(
                        f"模型 {model_gb:.1f} GB 超过当前空闲显存 {free_gb:.1f} GB，"
                        f"放不下的层会退回 CPU（速度骤降）。可调小 --n-gpu-layers 试试。")
        except Exception:
            pass

        task = _start_task("llama", cmd, {
            "port": port,
            "modelPath": meta["modelPath"],
            "engine": body.get("engine") or "llama",
            "exe": meta["exe"],
            "kv": f'{meta["kvK"]}/{meta["kvV"]}',
            "ctx": meta["ctx"],
        })
        return self._json({"taskId": task.id, "cmd": cmd, "port": port,
                           "warnings": warns, "caps": meta.get("caps")})

    # -- 生成测试 ---------------------------------------------------------- #
    @staticmethod
    def _probe(port) -> bool:
        """端口上有没有把服务起起来（/health 连得上就算）。"""
        return http_json(port, "/health", 3.0) is not None

    def _chat(self, body: dict):
        port = str(body.get("port") or 8421)
        payload = {
            "model": body.get("model") or "local",
            "messages": body.get("messages") or [],
            "max_tokens": int(body.get("maxTokens") or 512),
            "temperature": float(body.get("temperature") if body.get("temperature") is not None else 0.7),
        }
        if body.get("topP") not in (None, ""):
            payload["top_p"] = float(body["topP"])
        if body.get("topK") not in (None, ""):
            payload["top_k"] = int(body["topK"])
        if body.get("extraBody"):
            try:
                payload.update(json.loads(body["extraBody"]))
            except Exception:
                pass
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=float(body.get("timeout") or 600)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            return self._err(f"服务返回 HTTP {exc.code}: {detail}", 502)
        except Exception as exc:
            return self._err(f"连接失败: {exc}", 502)

        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        return self._json({
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish": ((data.get("choices") or [{}])[0]).get("finish_reason"),
            "usage": data.get("usage") or {},
            "latency": round(time.time() - t0, 2),
            "raw": json.dumps(data, ensure_ascii=False)[:4000],
        })


def _port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.4)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def _probe_tool(host: str, port: int) -> bool:
    """端口上是否已经跑着**本工具**（而不是别的程序）。"""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/env", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return isinstance(data, dict) and "python" in data and "drives" in data
    except Exception:
        return False


def _takeover_port(host: str, port: int) -> None:
    """启动前接管旧实例。

    http.server 默认开着 SO_REUSEADDR，在 Windows 上等于允许**第二个进程绑同一端口**，
    于是会出现「两个工具实例抢一个端口」：浏览器随机命中其中之一，页面数据自相矛盾
    （典型表现：一个窗口显存正常、另一个显示 0；刷新几次结果又变了）。
    这里在绑定前先请旧实例自己退出（走 /api/shutdown，它会顺带收掉子服务），
    拿不到响应才继续，尽量保证同一端口只有一个实例。
    """
    if not _probe_tool(host, port):
        return
    print(f"[i] 端口 {port} 上已有本工具在运行，正在接管（旧实例先退出）…")
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/api/shutdown", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=6).read()
    except Exception:
        pass
    for _ in range(30):
        if not _port_open(host, port):
            print("[i] 旧实例已退出，继续启动。")
            return
        time.sleep(0.3)
    print("[!] 旧实例没有退出，端口仍被占用。请手动关掉另一个工具箱窗口，"
          "否则两个实例会抢同一端口，页面数据会前后矛盾。")


def main() -> int:
    ap = argparse.ArgumentParser(description="FreeToken 转换工具")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(INDEX):
        print(f"[!] 找不到 index.html：{INDEX}", file=sys.stderr)
        return 1

    # 同一端口只允许一个实例：旧实例自己退出，避免 SO_REUSEADDR 下两个进程并存
    _takeover_port(args.host, args.port)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True

    # 进程退出（无论是 Ctrl+C、被 kill、还是用户点"退出工具")都先把托管的模型服务收掉，
    # "关掉工具 = 显存归还"，不留下占着显卡的孤儿进程。
    atexit.register(_shutdown_children)
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32            # type: ignore[attr-defined]

            def _ctrl_handler(ctrl_type):
                # 0=CTRL_C_EVENT, 2=CTRL_CLOSE_EVENT（关闭控制台窗口）
                if ctrl_type in (0, 2):
                    _shutdown_children()
                return 0

            win_handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(_ctrl_handler)
            kernel32.SetConsoleCtrlHandler(win_handler, 1)
        except Exception:
            pass

    # CPU 占用率是"两次采样取差值"，先跑起来，前端第一次请求就能拿到真数
    threading.Thread(target=_cpu_loop, daemon=True).start()
    # 显卡探测自检：32 位解释器是最常见的“有显卡却读不到”原因，启动时就说清楚。
    _bits = _host_bits()
    _smi = _nvidia_smi_path()
    _gpus = gpu_list()
    if _bits == 32:
        print(f"[!] 正在用 32 位 Python {sys.version.split()[0]} 运行（{sys.executable}）。")
        print("    32 位进程看不到 C:\\Windows\\System32 里的 nvidia-smi（会被重定向到 SysWOW64），")
        print("    显卡探测已自动改走 Sysnative 兼容通道。建议改用 64 位 Python 重新运行。")
    if _gpus:
        print(f"[i] 显卡：{_gpus[0].get('name')}"
              f"（显存 {(_gpus[0].get('totalMiB') or 0) / 1024:.0f} GB，"
              f"空闲 {(_gpus[0].get('freeMiB') or 0) / 1024:.1f} GB，来源 {_gpus[0].get('source')}）")
    else:
        print("[!] 未探测到 NVIDIA 显卡：nvidia-smi 与 NVML 都没返回数据。")
        print(f"    nvidia-smi 路径解析结果：{_smi or '一个都没找到'}")
        print("    请在命令行执行 nvidia-smi 确认驱动正常；若正常，多半是 Python 位数问题。")

    url = f"http://{args.host}:{args.port}/"
    print(f"FreeToken / llama.cpp 工具箱已启动 -> {url}")
    print("按 Ctrl+C 停止。")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出，清理托管的后台任务…")
    finally:
        _shutdown_children()
        print("已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
