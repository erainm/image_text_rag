# Created by erainm on 2025/11/25 13:58.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：memory_monitor
# @Description:内存监控

import psutil
import time
import threading
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    def __init__(self, warning_threshold_gb=26, critical_threshold_gb=24):
        self.warning_threshold = warning_threshold_gb * 1024 ** 3
        self.critical_threshold = critical_threshold_gb * 1024 ** 3
        self.is_monitoring = False
        self.monitor_thread = None

    def get_memory_info(self):
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }

    def print_memory_status(self):
        """打印内存状态"""
        mem_info = self.get_memory_info()
        status = f"内存使用: {mem_info['used'] / 1024 ** 3:.1f}GB / {mem_info['total'] / 1024 ** 3:.1f}GB ({mem_info['percent']:.1f}%)"

        if mem_info['available'] < self.critical_threshold:
            logger.warning("内存严重不足!")
        elif mem_info['available'] < self.warning_threshold:
            logger.warning("内存紧张，建议清理")

        return status

    def start_monitoring(self, interval=30):
        """开始内存监控"""
        self.is_monitoring = True

        def monitor_loop():
            while self.is_monitoring:
                status = self.print_memory_status()
                logger.info(status)
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止内存监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

from functools import wraps
# 内存保护装饰器
def memory_protected(func):
    """内存保护装饰器"""

    @wraps(func)  # 添加这行来保留原函数的元数据
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        mem_info = monitor.get_memory_info()

        if mem_info['available'] < 1 * 1024 ** 3:  # 小于1GB可用内存
            raise MemoryError("内存不足，无法执行操作")

        return func(*args, **kwargs)

    return wrapper