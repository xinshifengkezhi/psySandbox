import sys
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

#默认日志级别
DEFAULT_LEVEL = logging.INFO
#默认单个日志文件的大小(1mb)
DEFAULT_MAX_BYTES = 1024 * 1024



"""重写print方法，让我们每次输出时都向日志中写入一次输出内容"""
class DualOutput:
    def __init__(self, fileName):
        #保存原始的stdout，可以做到一边向控制台输出，一边写入日志
        self.terminal = sys.stdout
        self.logFile = open(fileName,'a',encoding='utf-8')

    """重写write方法"""
    def write(self,message):
        #将消息输出到控制台（保持原有的行为）
        self.terminal.write(message)
        #将同样的消息写入日志文件
        self.logFile.write(message)
        #立即刷新文件缓存区，确保消息及时写入磁盘
        self.logFile.flush()

    """重写flush方法，确保缓冲区的数据被立即写入"""
    def flush(self):
        self.terminal.flush()
        self.logFile.flush()

"""设置Python的标准logging模块"""
def setupLogging(logFile = 'output.log'):
    logFile = os.path.join('../logs', logFile)
    #配置logging模块的基本设置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            #将日志写入文件
            logging.FileHandler(logFile, encoding='utf-8'),
            #将日志输出到控制台
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

"""重定向所有的print语句到文件和控制台"""
def printToLog(logFile = 'output.log'):
    logFile = os.path.join('../logs', logFile)
    #创建DualOutput实例
    dual = DualOutput(logFile)

    sys.stdout = dual
    return dual

"""处理所有未捕获的异常"""
def handleExcept(excType, excValue, excTraceback):
    if issubclass(excType, KeyboardInterrupt):
        #不记录键盘中断
        sys.__excepthook__(excType, excValue, excTraceback)
        return
    logging.critical(
        """未捕获的异常""",
        exc_info=(excType, excValue, excTraceback)
    )


"""训练过程中的日志输出"""
def setup_logging(
        logDir: str, backupCount: int,
        level: Union[int, str] = DEFAULT_LEVEL,
        maxBytes: int = DEFAULT_MAX_BYTES,
) -> None:

    os.makedirs(logDir, exist_ok=True)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    #创建根日志记录器
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)
    if rootLogger.hasHandlers():
        rootLogger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    #--------------文件处理器（带轮转）--------------
    #按日期生成日志文件名
    logFile = os.path.join(logDir, f'{datetime.now().strftime("%Y%m%d")}.log')
    fileHandler = logging.handlers.RotatingFileHandler(
        logFile, maxBytes=maxBytes, backupCount=backupCount, encoding='utf-8'
    )
    fileHandler.setLevel(level)
    fileHandler.setFormatter(formatter)
    rootLogger.addHandler(fileHandler)

    # -------------- 控制台处理器（带轮转） --------------
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(level)
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)

    logging.captureWarnings(True)
    logging.getLogger('py.warnings').setLevel(logging.WARNING)

    logging.info('日志初始化完成，日志文件：%s', logFile)

def get_logger(name: str) -> logging.Logger:

    return logging.getLogger(name)

# -------------- 记录常见信息 --------------
def log_config(logger: logging.Logger, config: Dict[str, Any], title: str = "配置参数") -> None:
    """
    将配置字典记录到日志（INFO 级别）。
    """
    logger.info("=" * 50)
    logger.info(title)
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info("%-30s : %s", key, value)
    logger.info("=" * 50)


def log_data_info(
    logger: logging.Logger,
    data_name: str,
    num_samples: int,
    node_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    class_distribution: Optional[Dict[Any, int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    记录数据集的统计信息。
    """
    logger.info("----- %s 信息 -----", data_name)
    logger.info("样本数: %d", num_samples)
    if node_features is not None:
        logger.info("节点特征维度: %d", node_features)
    if num_classes is not None:
        logger.info("类别数: %d", num_classes)
    if class_distribution:
        logger.info("类别分布: %s", class_distribution)
    if extra:
        for k, v in extra.items():
            logger.info("%s: %s", k, v)
    logger.info("----------------------")


def log_data_split(
    logger: logging.Logger,
    kfold: int,
    train_size: int,
    val_size: int,
    test_size: int,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    stratify: Optional[str] = None,
) -> None:
    """
    记录数据划分信息。
    """
    logger.info(f"数据划分:第{kfold}折")
    logger.info("  训练集: %d 样本 (%.2f%%)", train_size, train_ratio * 100 if train_ratio else 0)
    logger.info("  验证集: %d 样本 (%.2f%%)", val_size, val_ratio * 100 if val_ratio else 0)
    logger.info("  测试集: %d 样本 (%.2f%%)", test_size, test_ratio * 100 if test_ratio else 0)
    if stratify:
        logger.info("  分层策略: %s", stratify)


def log_model_summary(
    logger: logging.Logger,
    kfold: int,
    model: Any,
    input_shape: Optional[tuple] = None,
    show_architecture: bool = True,
) -> None:
    """
    记录模型参数量、结构摘要等信息。
    注意：此函数需要模型实现 __str__ 或提供 summary() 方法。
    """
    logger.info(f"第{kfold}折模型信息")
    try:
        # 尝试获取总参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("模型总参数量: %d", total_params)
        logger.info("可训练参数量: %d", trainable_params)
        if input_shape:
            logger.info("期望输入形状: %s", input_shape)
        if show_architecture:
            logger.info("模型结构:\n%s", str(model))
    except Exception as e:
        logger.warning("无法完整记录模型信息: %s", e)


def log_test_results(
    logger: logging.Logger,
    kfold: int,
    metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    记录测试集上的评估结果。
    """
    logger.info("========== %s 测试集评估结果 ==========", f"第{kfold}折")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info("%-20s : %.6f", name, value)
        else:
            logger.info("%-20s : %s", name, value)
    if extra:
        for k, v in extra.items():
            logger.info("%-20s : %s", k, v)
    logger.info("===================================")
