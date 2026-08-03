"""
结构化日志配置。

- 服务端日志输出到 logs/server/ 目录
- 日志按天滚动，保留 30 天
- 控制台输出带颜色，方便调试
"""

import sys
from pathlib import Path
from loguru import logger

from config import settings


def setup_logger():
    """配置并返回 loguru logger 实例"""

    # 确保日志目录存在
    log_path = Path(settings.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 移除默认 handler
    logger.remove()

    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 文件输出（按天滚动，保留30天）
    logger.add(
        str(log_path / "server_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        compression="gz",
        backtrace=True,
        diagnose=True,
    )

    return logger


log = setup_logger()