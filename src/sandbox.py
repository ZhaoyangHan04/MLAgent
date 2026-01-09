import ast
import subprocess
import tempfile
import textwrap
import resource
import signal
import sys
from pathlib import Path
import re
import os

FORBIDDEN_MODULES = {
    "subprocess", "socket", "shutil", "inspect", "importlib",
}

FORBIDDEN_CALLS = {
    "exec", "eval", "compile", "__import__",
    # "open", "input",
}


class UnsafeCodeError(Exception):
    pass


def static_code_check(code: str):
    """
    Perform static AST-based safety checks.
    """
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # import os / subprocess ...
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    raise UnsafeCodeError(f"Forbidden import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_MODULES:
                raise UnsafeCodeError(f"Forbidden import from: {node.module}")

        # exec(), eval(), open(), etc.
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise UnsafeCodeError(f"Forbidden call: {node.func.id}")


def _limit_resources():
    """
    Limit CPU time and memory usage (Unix only).
    """
    # CPU time (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    # Max memory (512 MB)
    mem_bytes = 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


CODE_BLOCK_RE = re.compile(
    r"<code>\s*```python(.*?)```?\s*</code>",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract python code blocks from model output.
    """
    return [m.strip() for m in CODE_BLOCK_RE.findall(text)]


class CodeSandbox:
    def __init__(
        self,
        work_dir: str,
        timeout: int = 10,
        python_executable: str = sys.executable,
        tmp_dir: str = './tmp'
    ):
        self.work_dir = Path(work_dir).resolve()
        self.tmp_dir = Path(tmp_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.python = python_executable

    def run(self, code: str) -> str:
        """
        Safely execute Python code and return combined stdout/stderr.
        """
        # 1. Static safety check
        static_code_check(code)

        # 2. Wrap code with guards
        wrapped_code = (
            "import sys\n"
            "import traceback\n"
            "\n"
            "try:\n"
            + textwrap.indent(code, "    ")
            + "\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
        )
        # wrapped_code = (
        #     "import sys\n"
        #     "import traceback\n\n"
        #     "try:\n"
        #     + textwrap.indent(code, "    ")
        #     + "\n"
        #     "except Exception as e:\n"
        #     "    print(f'ERROR: {e}', file=sys.stderr)\n"
        #     "    traceback.print_exc(file=sys.stderr)\n\n"
        #     "# 强制刷新所有输出\n"
        #     "sys.stdout.flush()\n"
        #     "sys.stderr.flush()\n"
        # )

        with tempfile.TemporaryDirectory(dir = self.tmp_dir) as tmpdir:
            tmpdir = Path(tmpdir)

            script_path = tmpdir / "run.py"
            script_path.write_text(wrapped_code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [self.python, str(script_path)],
                    cwd=self.work_dir,               # restrict filesystem
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                    # preexec_fn=_limit_resources,     # CPU / memory limit
                    text=True,
                )
            except subprocess.TimeoutExpired:
                return "[Sandbox] Execution timed out."

        output = ""
        if proc.stdout:
            output += "<interpreter>" + "The code has run successfully. Code output: " + proc.stdout + "</interpreter>"
        if proc.stderr:
            output += "<interpreter>" + "The code didn't run successfully. [stderr]: " + proc.stderr + "</interpreter>"

        return output.strip() or "<interpreter>[Sandbox] No output.</interpreter>"

if __name__ == "__main__":
    sandbox = CodeSandbox(
        work_dir="/data0/hzy/DSAgent/data/openml_task_14970/dataset",
        timeout=10,
    )
    code = """
print(\"Testing with simpler model...\")\n\nimport os\nimport pandas as pd\nimport numpy as np\n\n# Load data\nbase_dir = \"/data0/hzy/DSAgent/data/openml_task_14970\"\nsplit_dir = os.path.join(base_dir, \"dataset\", \"splits\")\nfold_dir = os.path.join(split_dir, \"repeat_0_fold_0\")\n\ntrain_df = pd.read_csv(os.path.join(fold_dir, \"train.csv\"))\ntest_df = pd.read_csv(os.path.join(fold_dir, \"test.csv\"))\n\ntarget_col = 'Class'\nX_train = train_df.drop(columns=[target_col])\ny_train = train_df[target_col]\nX_test = test_df.drop(columns=[target_col])\ny_test = test_df[target_col]\n\nprint(f\"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}\")\n\n# Try a very simple model - DecisionTree with limited depth\ntry:\n    from sklearn.tree import DecisionTreeClassifier\n    from sklearn.metrics import accuracy_score, f1_score\n    \n    model = DecisionTreeClassifier(max_depth=5, random_state=42)\n    model.fit(X_train, y_train)\n    \n    y_pred = model.predict(X_test)\n    acc = accuracy_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred, average='macro')\n    \n    print(f\"\\nDecisionTree results:\")\n    print(f\"Accuracy: {acc:.4f}\")\n    print(f\"Macro-F1: {f1:.4f}\")\n    \nexcept Exception as e:\n    print(f\"Error with DecisionTree: {e}\")\n    \n    # Try even simpler - Dummy classifier\n    from sklearn.dummy import DummyClassifier\n    from sklearn.metrics import accuracy_score, f1_score\n    \n    dummy = DummyClassifier(strategy='most_frequent')\n    dummy.fit(X_train, y_train)\n    y_pred = dummy.predict(X_test)\n    acc = accuracy_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred, average='macro')\n    \n    print(f\"\\nDummy classifier (most frequent) results:\")\n    print(f\"Accuracy: {acc:.4f}\")\n    print(f\"Macro-F1: {f1:.4f}\")
"""
    print(sandbox.run(code))

# if __name__ == "__main__":
#     # 先测试一个简单的代码
#     sandbox = CodeSandbox(
#         work_dir="/data0/hzy/DSAgent/data/openml_task_14970/dataset",
#         timeout=10,
#     )
    
#     # 测试1: 最简单的代码
#     print("测试1: 简单打印")
#     simple_code = "print('Hello from sandbox')"
#     result = sandbox.run(simple_code)
#     print(f"结果: {result}\n")
    
#     # 测试2: 检查路径
#     print("测试2: 检查工作目录")
#     test_code = """
# import os
# print(f"当前工作目录: {os.getcwd()}")
# print(f"目录存在: {os.path.exists('.')}")
# """
#     result = sandbox.run(test_code)
#     print(f"结果: {result}\n")

#     # 测试3: 检查sklearn是否可用
#     print("测试3: 检查torch")
#     test_code = """
# try:
#     import torch
#     print(f"torch版本: {torch.__version__}")
# except ImportError as e:
#     print(f"无法导入torch: {e}")
# """

#     # 测试3: 检查sklearn是否可用
#     print("测试3: 检查sklearn")
#     test_code = """
# try:
#     import sklearn
#     print(f"sklearn版本: {sklearn.__version__}")
#     print("hello")
# except ImportError as e:
#     print(f"无法导入sklearn: {e}")
# """
#     result = sandbox.run(test_code)
#     print(f"结果: {result}\n")
