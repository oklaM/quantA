"""
日志工具配置
提供统一的日志接口
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import logging as log_config


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式"""

    # ANSI颜色码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record):
        # 添加颜色
        levelcolor = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{levelcolor}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    设置并返回一个logger

    Args:
        name: logger名称
        level: 日志级别
        log_file: 日志文件路径
        console: 是否输出到控制台

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 设置日志级别
    log_level = getattr(logging, (level or log_config.LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)

    # 日志格式
    formatter = logging.Formatter(fmt=log_config.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台处理器（带颜色）
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(log_config.FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_config.FILE_ENABLED:
        log_path = log_file or log_config.FILE_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=log_config.MAX_BYTES,
            backupCount=log_config.BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 模块级logger
root_logger = setup_logger("quantA")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger

    Args:
        name: logger名称（通常使用__name__）

    Returns:
        logger实例
    """
    return setup_logger(f"quantA.{name}")


# 便捷函数
def debug(msg, *args, **kwargs):
    root_logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    root_logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    root_logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    root_logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    root_logger.critical(msg, *args, **kwargs)


__all__ = [
    "setup_logger",
    "get_logger",
    "logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]

# 导出默认logger实例
logger = root_logger
