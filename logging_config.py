"""
BindCraft 日志系统配置模块
提供统一的日志配置功能
"""

import logging
import os
from datetime import datetime


def setup_logging(design_path=None):
    """
    设置完善的日志系统
    
    Args:
        design_path (str, optional): 设计输出路径，用于保存日志文件
    
    Returns:
        logging.Logger: 配置好的日志记录器
    
    日志级别说明:
        - INFO级别：一般信息，进度更新
        - WARNING级别：警告信息，非致命错误  
        - ERROR级别：错误信息，但程序可以继续
        - DEBUG级别：详细的调试信息
    """
    # 创建日志记录器
    logger = logging.getLogger('BindCraft')
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器（避免重复）
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器（显示INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（记录所有级别）
    if design_path:
        log_file = os.path.join(design_path, f'bindcraft_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        os.makedirs(design_path, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger
