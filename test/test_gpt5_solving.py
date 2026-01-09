# import pandas as pd
# from pathlib import Path

# fold_dir = Path("data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0")

# train_df = pd.read_csv(fold_dir / "train.csv")
# test_df  = pd.read_csv(fold_dir / "test.csv")

# print("=== SHAPES ===")
# print("Train:", train_df.shape)
# print("Test :", test_df.shape)

# print("\n=== TARGET DISTRIBUTION (TRAIN) ===")
# print(train_df["class"].value_counts())
# print("\n=== TARGET DISTRIBUTION (TEST) ===")
# print(test_df["class"].value_counts())


# import pandas as pd
# from pathlib import Path

# train_df = pd.read_csv(
#     Path("data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/train.csv")
# )

# print("=== DTYPE COUNTS ===")
# print(train_df.dtypes.value_counts())

# print("\n=== DTYPE PER COLUMN ===")
# print(train_df.dtypes)


# print("\n=== TOP-15 MISSING RATIO (TRAIN) ===")
# missing_ratio = train_df.isna().mean().sort_values(ascending=False)
# print(missing_ratio.head(15))

# print("\n=== NOMINAL CARDINALITY (TOP-10) ===")

# nominal_cols = train_df.select_dtypes(include=["object"]).columns
# cardinality = {
#     col: train_df[col].nunique(dropna=True)
#     for col in nominal_cols
# }

# for col, n in sorted(cardinality.items(), key=lambda x: -x[1])[:10]:
#     print(col, ":", n)

# import pandas as pd
# from pathlib import Path

# train_path = Path(
#     "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/train.csv"
# )
# test_path = Path(
#     "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/test.csv"
# )

# DROP_COLS = [
#     "p", "s", "jurofm", "corr", "marvi", "m",
#     "bc", "exptl", "blue%2Fbright%2Fvarn%2Fclean",
#     "phos", "surface-finish", "packing"
# ]

# train_df = pd.read_csv(train_path)
# test_df  = pd.read_csv(test_path)

# train_df_clean = train_df.drop(columns=DROP_COLS)
# test_df_clean  = test_df.drop(columns=DROP_COLS)

# print("=== SHAPES AFTER DROP ===")
# print("Train:", train_df_clean.shape)
# print("Test :", test_df_clean.shape)

# print("\n=== ANY ALL-NA COLUMN LEFT? ===")
# print(train_df_clean.isna().mean().sort_values(ascending=False).head(10))

# print("\n=== DTYPES AFTER DROP ===")
# print(train_df_clean.dtypes.value_counts())


# import pandas as pd
# from pathlib import Path

# train_path = Path(
#     "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/train.csv"
# )
# test_path = Path(
#     "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/test.csv"
# )

# # 上一步的 hard drop
# DROP_COLS_STAGE1 = [
#     "p", "s", "jurofm", "corr", "marvi", "m",
#     "bc", "exptl", "blue%2Fbright%2Fvarn%2Fclean",
#     "phos", "surface-finish", "packing"
# ]

# train_df = pd.read_csv(train_path).drop(columns=DROP_COLS_STAGE1)
# test_df  = pd.read_csv(test_path).drop(columns=DROP_COLS_STAGE1)

# # 计算缺失率
# missing_ratio = train_df.isna().mean()

# # 第二轮 drop（>=95% 缺失）
# DROP_COLS_STAGE2 = missing_ratio[missing_ratio >= 0.95].index.tolist()

# print("=== STAGE-2 DROP COLS (>=95% missing) ===")
# print(DROP_COLS_STAGE2)

# train_df2 = train_df.drop(columns=DROP_COLS_STAGE2)
# test_df2  = test_df.drop(columns=DROP_COLS_STAGE2)

# print("\n=== SHAPES AFTER STAGE-2 DROP ===")
# print("Train:", train_df2.shape)
# print("Test :", test_df2.shape)

# print("\n=== TOP-10 MISSING AFTER STAGE-2 ===")
# print(train_df2.isna().mean().sort_values(ascending=False).head(10))

# print("\n=== DTYPES AFTER STAGE-2 ===")
# print(train_df2.dtypes.value_counts())


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

DROP_COLS = [
    # stage-1
    "p", "s", "jurofm", "corr", "marvi", "m",
    "bc", "exptl", "blue%2Fbright%2Fvarn%2Fclean",
    "phos", "surface-finish", "packing",
    # stage-2
    "enamelability", "chrom", "ferro",
]

class AnnealFinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.num_cols_ = None
        self.cat_cols_ = None
        self.medians_ = {}

    def fit(self, X, y=None):
        X = X.drop(columns=DROP_COLS, errors="ignore")

        self.num_cols_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()

        for col in self.num_cols_:
            self.medians_[col] = X[col].median()

        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=DROP_COLS, errors="ignore")

        # numeric
        for col in self.num_cols_:
            X[col] = X[col].fillna(self.medians_[col])

        # categorical
        for col in self.cat_cols_:
            X[col] = X[col].fillna("__MISSING__")

        return X

import pandas as pd
from pathlib import Path

train_path = Path(
    "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/train.csv"
)
test_path = Path(
    "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0/test.csv"
)

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET_COL = "class"

X_train = train_df.drop(columns=[TARGET_COL])
X_test  = test_df.drop(columns=[TARGET_COL])

prep = AnnealFinalPreprocessor(target_col=TARGET_COL)
prep.fit(X_train)

Xt_train = prep.transform(X_train)
Xt_test  = prep.transform(X_test)

print("=== SHAPES AFTER PREPROCESS ===")
print("Train:", Xt_train.shape)
print("Test :", Xt_test.shape)

print("\n=== ANY NA LEFT? ===")
print(Xt_train.isna().any().any(), Xt_test.isna().any().any())

print("\n=== DTYPES AFTER PREPROCESS ===")
print(Xt_train.dtypes.value_counts())

print("\n=== SAMPLE ROW ===")
print(Xt_train.head(3))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path

TARGET_COL = "class"

pipeline = Pipeline([
    ("prep", AnnealFinalPreprocessor(target_col=TARGET_COL)),
    ("encode", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])


fold_dir = Path(
    "data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_0"
)

train_df = pd.read_csv(fold_dir / "train.csv")
test_df  = pd.read_csv(fold_dir / "test.csv")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=== SINGLE FOLD RESULT ===")
print("Accuracy:", acc)

from sklearn.metrics import confusion_matrix
import pandas as pd

labels = sorted(y_test.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("=== CONFUSION MATRIX (FOLD 0) ===")
print(cm_df)

pred_df = test_df.copy()
pred_df["pred"] = y_pred

print("=== TEST PREDICTIONS SAMPLE ===")
print(pred_df[[TARGET_COL, "pred"]].head(15))

from sklearn.metrics import accuracy_score

accs = []

for fold in range(10):
    fold_dir = Path(
        f"data/tasks/classification/openml_task_2/dataset/splits/repeat_0_fold_{fold}"
    )

    train_df = pd.read_csv(fold_dir / "train.csv")
    test_df  = pd.read_csv(fold_dir / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)

    print(f"Fold {fold}: acc={acc:.4f}")

print("\n=== CV SUMMARY ===")
print(f"Mean acc: {sum(accs)/len(accs):.4f}")
print(f"Std  acc: {pd.Series(accs).std():.4f}")
