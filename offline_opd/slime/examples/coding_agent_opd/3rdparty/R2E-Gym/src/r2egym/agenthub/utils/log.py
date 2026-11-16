import logging
import os
from rich.logging import RichHandler


# 集中控制 R2E-Gym 自身日志级别的环境变量。所有 R2E-Gym 模块的 logger 都经由本
# 函数创建（agenthub.utils.utils.get_logger 只是再导出本函数），因此在这里设一次
# 即可静默/放开 R2E-Gym 全部日志，无需改每个文件头部。
#   R2EGYM_LOG_LEVEL=ERROR  -> 仅 ERROR/CRITICAL（默认，静默 INFO/WARNING/DEBUG）
#   R2EGYM_LOG_LEVEL=INFO   -> 放开到 INFO 便于调试
# 取值为标准 logging 级别名（大小写不敏感）；非法值回退到调用方传入的 level。
_R2EGYM_LOG_LEVEL = os.getenv("R2EGYM_LOG_LEVEL", "ERROR")


def get_logger(
    name: str = "defaultLogger", level: int = logging.INFO
) -> logging.Logger:
    """
    Returns a logger configured with RichHandler, ensuring that
    any existing handlers on this logger are removed first.

    :param name: Name of the logger.
    :param level: Logging level.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)

    # Remove any existing handlers from this logger
    # (the loop approach or just `logger.handlers = []`)
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # NOTE: 绝不清空 root logger 的 handler。之前这里会 removeHandler(root ...)，
    # 会误杀外部框架（如 slime / examples.coding_agent_opd.generate）挂在 root 上
    # 的 handler，导致其 logger.info 全部无输出。R2E-Gym 与外部日志体系通过
    # propagate=False 解耦，无需动 root。

    # 环境变量优先，集中设定 R2E-Gym 日志级别；未设置/非法时回退到调用方 level。
    env_level = logging.getLevelName(_R2EGYM_LOG_LEVEL.upper())
    if isinstance(env_level, int):
        level = env_level
    logger.setLevel(level)

    # 不向 root 冒泡：R2E-Gym 日志只走自己的 RichHandler，既与外部日志解耦，也避免
    # 在 root 保留 handler 后出现重复输出。
    logger.propagate = False

    # If logging is globally disabled, do not add any handlers
    if not logging.getLogger().isEnabledFor(logging.CRITICAL):
        return logger

    # Create a RichHandler
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False)

    # Set the format for the logger
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    # Add the RichHandler to the logger
    logger.addHandler(rich_handler)

    # add File Handler if R2EGym_LOG_FILE environment variable exists
    log_file_path = os.getenv('R2EGym_LOG_FILE')
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
