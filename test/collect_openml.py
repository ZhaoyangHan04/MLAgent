import json
import openml
import pandas as pd
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path("./data/tasks")
TASKS_PER_TYPE = 500

from openml.tasks import TaskType

TASK_TYPE_MAP = {
    # "classification": TaskType.SUPERVISED_CLASSIFICATION,
    "regression": TaskType.SUPERVISED_REGRESSION,
    "clustering": TaskType.CLUSTERING,
    "learning_curve": TaskType.LEARNING_CURVE,
}



# ---------- basic utils ----------

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_task_dirs(task_dir: Path):
    (task_dir / "task").mkdir(parents=True, exist_ok=True)
    (task_dir / "dataset" / "splits").mkdir(parents=True, exist_ok=True)


# ---------- meta extraction ----------

def extract_task_meta(task):
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "target_feature": task.target_name,
        "estimation_procedure": task.estimation_procedure,
        "evaluation_metric": task.evaluation_measure,
        "dataset_id": task.dataset_id,
    }


def extract_dataset_meta(dataset):
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "version": dataset.version,
        "default_target_attribute": dataset.default_target_attribute,
        "qualities": dataset.qualities,
    }


def extract_features_meta(dataset):
    features = []
    for _, feat in dataset.features.items():
        features.append({
            "index": feat.index,
            "name": feat.name,
            "data_type": feat.data_type,
            "is_target": feat.name == dataset.default_target_attribute,
            "is_nominal": feat.data_type == "nominal",
        })
    return features


# ---------- core logic per task ----------

def export_single_task(task_id: int, task_root: Path):
    task = openml.tasks.get_task(task_id)
    task_dir = task_root / f"openml_task_{task_id}"
    ensure_task_dirs(task_dir)

    # task meta
    save_json(
        extract_task_meta(task),
        task_dir / "task" / "task_meta.json"
    )

    # dataset
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=task.target_name
    )

    df = X.copy()
    if y is not None:
        df[task.target_name] = y

    df.to_csv(task_dir / "dataset" / "full.csv", index=False)

    # dataset meta
    save_json(
        extract_dataset_meta(dataset),
        task_dir / "dataset" / "dataset_meta.json"
    )

    save_json(
        extract_features_meta(dataset),
        task_dir / "dataset" / "features.json"
    )

    # splits (if exist)
    try:
        n_repeats, n_folds, _ = task.get_split_dimensions()
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=repeat,
                    fold=fold
                )

                fold_dir = (
                    task_dir
                    / "dataset"
                    / "splits"
                    / f"repeat_{repeat}_fold_{fold}"
                )
                fold_dir.mkdir(parents=True, exist_ok=True)

                df.iloc[train_idx].to_csv(fold_dir / "train.csv", index=False)
                df.iloc[test_idx].to_csv(fold_dir / "test.csv", index=False)

    except Exception:
        # clustering / learning curve 等 task 没有 train-test split
        pass


# ---------- batch collection ----------

def collect_tasks_for_type(type_key: str, openml_type: str):
    print(f"\n[INFO] Collecting {TASKS_PER_TYPE} tasks for {type_key}")

    task_root = BASE_DIR / type_key
    task_root.mkdir(parents=True, exist_ok=True)

    task_list = openml.tasks.list_tasks(
        task_type=openml_type,
        output_format="dataframe"
    )

    task_ids = task_list["tid"].tolist()[:TASKS_PER_TYPE]

    for task_id in tqdm(task_ids, desc=f"{type_key}"):
        try:
            export_single_task(task_id, task_root)
        except Exception as e:
            # 保证 pipeline 不中断
            print(f"[WARN] Skip task {task_id}: {e}")


# ---------- entry ----------

def run_collection():
    for type_key, openml_type in TASK_TYPE_MAP.items():
        collect_tasks_for_type(type_key, openml_type)


if __name__ == "__main__":
    run_collection()
