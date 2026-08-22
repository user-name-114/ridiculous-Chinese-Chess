import platform
import os
import sys

print("=" * 60)
print("系统信息")
print("=" * 60)
print(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")
print(f"机器架构: {platform.machine()}")
print(f"处理器标识: {platform.processor()}")
print(f"Python: {sys.version.split()[0]}")

print()
print("=" * 60)
print("CPU")
print("=" * 60)
print(f"逻辑核心数: {os.cpu_count()}")

# 尝试用 psutil 获取更详细信息
try:
    import psutil
    print(f"物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"CPU 频率: {psutil.cpu_freq().max:.0f} MHz (max)")
    print(f"CPU 使用率: {psutil.cpu_percent(interval=1)}%")

    print()
    print("=" * 60)
    print("内存")
    print("=" * 60)
    mem = psutil.virtual_memory()
    print(f"总内存: {mem.total / 1024**3:.1f} GB")
    print(f"可用内存: {mem.available / 1024**3:.1f} GB")

    print()
    print("=" * 60)
    print("磁盘")
    print("=" * 60)
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            print(f"{part.device} ({part.mountpoint}): 总 {usage.total/1024**3:.1f} GB, 可用 {usage.free/1024**3:.1f} GB")
        except Exception:
            pass
except ImportError:
    print("(psutil 未安装，用 subprocess 查 wmic)")

# 用 Windows 命令查 CPU 型号
print()
print("=" * 60)
print("CPU 型号（wmic）")
print("=" * 60)
import subprocess
try:
    r = subprocess.run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:list"],
                       capture_output=True, text=True, timeout=30)
    print(r.stdout.strip())
except Exception as e:
    print(f"wmic 查询失败: {e}")

# 查内存（wmic）
print()
print("=" * 60)
print("内存（wmic）")
print("=" * 60)
try:
    r = subprocess.run(["wmic", "memorychip", "get", "Capacity,Speed", "/format:list"],
                       capture_output=True, text=True, timeout=30)
    # 汇总容量
    total = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Capacity="):
            try:
                total += int(line.split("=")[1])
            except ValueError:
                pass
    if total > 0:
        print(f"总内存容量: {total / 1024**3:.1f} GB")
    print(r.stdout.strip())
except Exception as e:
    print(f"wmic 查询失败: {e}")

# 查 GPU
print()
print("=" * 60)
print("GPU")
print("=" * 60)
try:
    r = subprocess.run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:list"],
                       capture_output=True, text=True, timeout=30)
    print(r.stdout.strip())
except Exception as e:
    print(f"wmic 查询失败: {e}")
