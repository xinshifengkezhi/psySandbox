"""
    定义注册表来调用各个模块
"""
operations = {}

def register(name):
    """注册函数用的装饰器，代表的函数已经在自定义配置文件里写清了"""
    def decorator(func):
        operations[name] = func
        return func
    return decorator
