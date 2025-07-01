import time
import json
import threading
import requests
import logging
import sys
from server import PromptServer

# 尝试截获和过滤HiDream日志
class HiDreamFilter(logging.Filter):
    def filter(self, record):
        # 如果日志消息来自HiDream并包含特定关键词，过滤掉
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if record.msg.startswith("HiDream:") and (
                "unloading all models" in record.msg or 
                "Cleaning up all cached models" in record.msg or
                "Cache cleared" in record.msg
            ):
                return False
        return True

# 应用日志过滤器
for handler in logging.root.handlers:
    handler.addFilter(HiDreamFilter())

class MemoryManager:
    def __init__(self):
        self.server = PromptServer.instance
        self.enabled = False
        self.interval = 60  # 默认60秒
        self.timer = None
        self.timer_lock = threading.Lock()
        self.verbose = False  # 默认不输出详细信息
        self.last_free_time = 0  # 上次释放内存的时间
        self.min_interval = 0.1  # 最小间隔时间（秒）- 改为0.1秒
        
    def start(self):
        """启动内存清理定时器"""
        with self.timer_lock:
            if self.timer is not None:
                self.stop()
            
            self.enabled = True
            self.timer = threading.Timer(self.interval, self._timer_callback)
            self.timer.daemon = True
            self.timer.start()
            # 仅在首次启动时输出一条消息
            if self.verbose:
                print(f"内存管理器已启动，间隔: {self.interval}秒")
    
    def stop(self):
        """停止内存清理定时器"""
        with self.timer_lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
                self.enabled = False
                if self.verbose:
                    print("内存管理器已停止")
    
    def _timer_callback(self):
        """定时器回调函数，执行内存清理并重新设置定时器"""
        self.free_memory()
        
        # 重新设置定时器以实现循环调用
        if self.enabled:
            with self.timer_lock:
                self.timer = threading.Timer(self.interval, self._timer_callback)
                self.timer.daemon = True
                self.timer.start()
    
    def free_memory(self):
        """调用API释放内存"""
        try:
            # 检查是否已经过了最小间隔时间
            current_time = time.time()
            if current_time - self.last_free_time < self.min_interval:
                if self.verbose:
                    print(f"内存释放太频繁，跳过此次释放")
                return {"status": "skipped", "reason": "too frequent"}
            
            self.last_free_time = current_time
            
            # 使用静默方式请求内存释放
            with open("/dev/null", "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                
                # 暂时重定向标准输出和错误输出
                if not self.verbose:
                    sys.stdout = devnull
                    sys.stderr = devnull
                
                try:
                    # 直接使用请求调用ComfyUI的API
                    url = "http://127.0.0.1:8188/free"
                    payload = {
                        "unload_models": True, 
                        "free_memory": True
                    }
                    response = requests.post(url, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        result = {"error": f"状态码: {response.status_code}"}
                        
                    return result
                finally:
                    # 恢复标准输出和错误输出
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
        except Exception as e:
            if self.verbose:
                print(f"释放内存出错: {str(e)}")
            return {"error": str(e)}

# 创建全局实例
memory_manager = MemoryManager()

class MemoryManagerNode:
    """
    内存管理器节点
    周期性调用API释放内存
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "interval_seconds": ("FLOAT", {"default": 60, "min": 0.1, "max": 60, "step": 0.1}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("状态",)
    FUNCTION = "manage_memory"
    CATEGORY = "🐳Pond/Tools"
    
    def manage_memory(self, enabled, interval_seconds, verbose):
        global memory_manager
        
        # 更新设置
        memory_manager.interval = interval_seconds
        memory_manager.verbose = verbose
        
        # 根据设置启动或停止
        if enabled and not memory_manager.enabled:
            memory_manager.start()
            status = f"内存管理器已启动，间隔: {interval_seconds}秒"
        elif not enabled and memory_manager.enabled:
            memory_manager.stop()
            status = "内存管理器已停止"
        elif enabled and memory_manager.enabled and memory_manager.interval != interval_seconds:
            # 如果间隔变化，重启定时器
            memory_manager.stop()
            memory_manager.interval = interval_seconds
            memory_manager.start()
            status = f"内存管理器已更新，间隔: {interval_seconds}秒"
        elif enabled:
            status = f"内存管理器正在运行，间隔: {interval_seconds}秒"
        else:
            status = "内存管理器已停止"
            
        return (status,)
    
    @classmethod
    def IS_CHANGED(cls, enabled, interval_seconds, verbose):
        # 设置更改后重新执行节点
        global memory_manager
        if enabled != memory_manager.enabled or interval_seconds != memory_manager.interval or verbose != memory_manager.verbose:
            return True
        return False

# 在脚本结束时停止定时器
import atexit

@atexit.register
def cleanup():
    global memory_manager
    if memory_manager.enabled:
        memory_manager.stop()

# 注册节点
NODE_CLASS_MAPPINGS = {
    "内存管理器": MemoryManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "内存管理器": "🐳内存管理器"
}