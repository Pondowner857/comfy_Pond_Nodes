import asyncio
import threading
import os
import time
import psutil
import torch
import server
import sys
from server import PromptServer

# 尝试导入NVIDIA GPU监控库
try:
    import pynvml
    PYNVML_LOADED = True
    pynvml.nvmlInit()
except:
    PYNVML_LOADED = False

# 监控线程锁
monitor_lock = threading.Lock()

# 获取命令行参数中指定的GPU
def get_cuda_device_indices():
    try:
        cuda_indices = []
        for i, arg in enumerate(sys.argv):
            if arg == '--cuda-device' and i + 1 < len(sys.argv):
                device_arg = sys.argv[i + 1]
                cuda_indices = [int(idx.strip()) for idx in device_arg.split(',')]
                return cuda_indices
        return []
    except Exception as e:
        return []

# 获取命令行指定的CUDA设备
CUDA_DEVICE_INDICES = get_cuda_device_indices()

class HardwareInfo:
    """硬件信息获取类"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.monitored_indices = self._get_monitored_indices()
        self.last_gpu_info = []  # 缓存上一次的GPU信息
    
    def _get_monitored_indices(self):
        """确定要监控的GPU索引"""
        # 如果命令行指定了GPU，优先使用命令行指定的
        if CUDA_DEVICE_INDICES:
            # 不再检查物理设备范围，即使指定的索引超出实际设备数量也监控它
            # 这样可以支持将GPU 0的信息显示为GPU 1
            return CUDA_DEVICE_INDICES
        
        # 否则，监控所有可用的GPU
        return list(range(self.device_count))

    def get_gpu_info(self):
        """获取所有GPU信息（带错误处理和缓存）"""
        if not self.cuda_available or not PYNVML_LOADED:
            return self.last_gpu_info  # 返回缓存数据

        try:
            gpus = []
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # 获取GPU利用率
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    except:
                        utilization = 0

                    # 获取GPU温度
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
                    except:
                        temperature = 0

                    # 获取显存信息
                    try:
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_total = memory.total
                        vram_used = memory.used
                        vram_used_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                    except:
                        vram_total = 0
                        vram_used = 0
                        vram_used_percent = 0

                    # 获取GPU名称
                    try:
                        gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    except:
                        gpu_name = f"GPU {i}"

                    gpus.append({
                        'index': i,
                        'name': gpu_name,
                        'gpu_utilization': utilization,
                        'gpu_temperature': temperature,
                        'vram_total': vram_total,
                        'vram_used': vram_used,
                        'vram_used_percent': vram_used_percent,
                        'is_cmdline_specified': i in self.monitored_indices
                    })
                except Exception:
                    continue

            # 更新缓存
            self.last_gpu_info = gpus
            return gpus

        except Exception:
            # 出错时返回缓存数据
            return self.last_gpu_info
    
    def get_status(self):
        """获取硬件状态"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        ram_total = memory.total
        ram_used = memory.used
        ram_used_percent = memory.percent
        
        # GPU信息
        gpus = self.get_gpu_info()
        
        return {
            'cpu_utilization': cpu_percent,
            'ram_total': ram_total,
            'ram_used': ram_used,
            'ram_used_percent': ram_used_percent,
            'device_type': 'cuda' if self.cuda_available else 'cpu',
            'gpus': gpus
        }

class HardwareMonitor:
    """硬件监控类"""
    
    def __init__(self, update_interval=1):
        self.update_interval = update_interval  # 更新间隔（秒）
        self.hardware_info = HardwareInfo()
        self.monitor_thread = None
        self.thread_controller = threading.Event()
        
        # 启动监控线程
        self.start_monitor()
    
    async def send_message(self, data):
        """发送监控数据到WebSocket"""
        # 使用ComfyUI服务器发送数据
        PromptServer.instance.send_sync('hardware.monitor', data)
    
    def start_monitor_loop(self):
        """启动监控循环"""
        # 静默模式，不输出启动信息
        asyncio.run(self.monitor_loop())
    
    async def monitor_loop(self):
        """监控循环"""
        while not self.thread_controller.is_set():
            # 获取硬件状态
            status = self.hardware_info.get_status()
            
            # 发送数据到WebSocket
            await self.send_message(status)
            
            # 等待指定的间隔时间
            await asyncio.sleep(self.update_interval)
    
    def start_monitor(self):
        """开始监控"""
        # 如果线程已存在，先停止
        if self.monitor_thread is not None:
            self.stop_monitor()
        
        # 清除控制信号
        self.thread_controller.clear()
        
        # 启动新线程
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            monitor_lock.acquire()
            self.monitor_thread = threading.Thread(target=self.start_monitor_loop)
            monitor_lock.release()
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitor(self):
        """停止监控"""
        self.thread_controller.set()

# 创建监控实例
hardware_monitor = HardwareMonitor(update_interval=5)

# 添加API端点
@PromptServer.instance.routes.get("/api/hardware/monitor/switch")
async def monitor_switch(request):
    """开启/关闭监控"""
    data = await request.json()
    if 'enabled' in data:
        if data['enabled']:
            hardware_monitor.start_monitor()
        else:
            hardware_monitor.stop_monitor()
    return server.web.json_response({"success": True})

# 静默初始化，不输出日志