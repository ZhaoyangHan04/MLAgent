# 在你的主程序（不是沙盒）中运行
import sys
import subprocess

print(f"当前Python: {sys.executable}")
print(f"Python版本: {sys.version}")

# 检查sklearn
try:
    import sklearn
    print(f"sklearn版本: {sklearn.__version__}")
except ImportError:
    print("sklearn未安装")

# 检查pandas
try:
    import pandas
    print(f"pandas版本: {pandas.__version__}")
except ImportError:
    print("pandas未安装")