import os
import json
import time
from typing import Dict, Any, List

import logging
log = logging.getLogger(__name__)

# === 你已有的组件 ===
from api_caller import ReActTrajectoryCaller
from sandbox import CodeSandbox, extract_code_blocks
from prompts import REACT_ML_ROLLOUT_SYSTEM_PROMPT, REACT_ML_ROLLOUT_TASK_PROMPT

MAX_STEPS = 20  # 防止死循环


class TrajectoryRunner:
    """
    End-to-end runner that:
    - drives multi-round ReAct dialogue
    - executes code safely
    - feeds execution results back
    """

    def __init__(
        self,
        caller: ReActTrajectoryCaller,
        sandbox: CodeSandbox,
        max_steps: int = MAX_STEPS,
    ):
        self.caller = caller
        self.sandbox = sandbox
        self.max_steps = max_steps

        # 用于持久化 trajectory
        self.trajectory: List[Dict[str, Any]] = []

    def _record(self, role: str, content: str):
        self.trajectory.append(
            {
                "role": role,
                "content": content,
            }
        )

    def run(self) -> Dict[str, Any]:
        """
        Run the ReAct loop until <answer> is produced or step limit reached.
        """
        self._record("system", self.caller.system_prompt)
        self._record("user", self.caller.task_prompt)

        for step_idx in range(self.max_steps):
            log.info(f"[Step {step_idx+1}/{self.max_steps}] Requesting model response")
            # === Step 1: Ask model to continue ===
            assistant_reply = self.caller.step()
            self._record("assistant", assistant_reply)

            # === Termination check ===
            if "<answer>" in assistant_reply:
                return {
                    "status": "finished",
                    "steps": step_idx + 1,
                    "trajectory": self.trajectory,
                }
                log.info(f"[Step {step_idx+1}] <answer> detected, finishing trajectory")


            log.info(
                f"[Step {step_idx+1}] Model replied | "
                f"length={len(assistant_reply)} | "
                f"has_code={'<code>' in assistant_reply} | "
                f"has_answer={'<answer>' in assistant_reply}"
            )

            # === Step 2: Extract and execute code ===
            code_blocks = extract_code_blocks(assistant_reply)

            if not code_blocks:
                # 没有 code，继续让模型 reasoning
                continue

            for code in code_blocks:
                try:
                    exec_output = self.sandbox.run(code)
                except Exception as e:
                    exec_output = f"[Sandbox Error] {e}"
                    log.error(
                        f"[Step {step_idx+1}] Sandbox execution failed: {e}",
                        exc_info=True,
                    )


                log.info(
                    f"[Step {step_idx+1}] Executing code block | chars={len(code)}"
                )

                # === Step 3: Feed execution result back ===
                self.caller.add_interpreter(exec_output)
                self._record("interpreter", exec_output)

            # 给模型一点“时间感”
            time.sleep(0.5)

        log.warning(
            f"Max steps reached ({self.max_steps}), terminating trajectory"
        )

        return {
            "status": "max_steps_exceeded",
            "steps": self.max_steps,
            "trajectory": self.trajectory,
        }

def run_single_task(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run trajectory generation for ONE ML task.
    """
    caller = ReActTrajectoryCaller(
        model="deepseek-v3.2",
        api_base="http://123.129.219.111:3000/v1",
        temperature=0.2,
        max_tokens=4096,
    )

    # system prompt（最小但稳定）
    caller.add_system(
        REACT_ML_ROLLOUT_SYSTEM_PROMPT
    )

    # rollout prompt（f-string 注入）
    rollout_prompt = REACT_ML_ROLLOUT_TASK_PROMPT.format(**task_cfg)
    caller.add_user(rollout_prompt)

    sandbox = CodeSandbox(
        work_dir=task_cfg["dataset_root_dir"],
        timeout=30,
    )

    log.info(f"CodeSandbox initialized | work_dir={sandbox.work_dir}")

    runner = TrajectoryRunner(
        caller=caller,
        sandbox=sandbox,
        max_steps=20,
    )

    result = runner.run()
    return result

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Define the base directory for the task
    task_base_dir = "/data0/hzy/DSAgent/data/openml_task_14970"

    # Derive other paths based on the base directory
    dataset_dir = os.path.join(task_base_dir, "dataset")
    splits_dir = os.path.join(dataset_dir, "splits")
    task_dir = os.path.join(task_base_dir, "task")

    # Load the task meta data from the task_meta.json file
    task_meta_file = os.path.join(task_dir, "task_meta.json")
    with open(task_meta_file, "r") as f:
        task_meta = json.load(f)

    # Function to generate the dataset tree dynamically
    def generate_dataset_tree(dataset_dir, splits_dir):
        # List of files in the dataset directory (excluding the splits directory)
        dataset_files = ["full.csv", "features.json", "dataset_meta.json"]
        dataset_tree = "dataset/\n"
        
        # Add the dataset files to the tree
        for file in dataset_files:
            dataset_tree += f"  {file}\n"

        # Add the splits directory and its files
        dataset_tree += "  splits/\n"
        for root, dirs, files in os.walk(splits_dir):
            # Process files in each fold directory (train and test)
            for file in files:
                if file.endswith(".csv"):  # Only interested in .csv files
                    relative_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    dataset_tree += f"    {relative_path}\n"
                    
        return dataset_tree

    # Generate the dynamic dataset tree
    dataset_tree = generate_dataset_tree(dataset_dir, splits_dir)

    # Define the task configuration
    task_config = {
        "dataset_root_dir": dataset_dir,
        "task_meta": task_meta,  # Directly load from task_meta.json
        "dataset_tree": dataset_tree,
        "file_dir": task_base_dir,
    }

    log.info(
        f"Starting trajectory generation | task_id={task_config.get('task_meta').get('task_id')} "
        f"| dataset_root_dir={task_config.get('dataset_root_dir')} "
        f"| Whole task config: {task_config}"
    )

    trj = run_single_task(task_config)

    print(f"Trajectory status: {trj['status']}")
    print(f"Total steps: {trj['steps']}")

    # 保存为 JSON（或 JSONL）
    with open("trajectory_14970_4.json", "w", encoding="utf-8") as f:
        json.dump(trj, f, indent=2, ensure_ascii=False)
