import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_data(base_dir="/data0/hzy/DSAgent/data/openml_task_14970", fold_idx=0):
    """加载训练和测试数据"""
    split_dir = os.path.join(base_dir, "dataset", "splits")
    fold_dir = os.path.join(split_dir, f"repeat_0_fold_{fold_idx}")
    
    # 加载数据
    train_df = pd.read_csv(os.path.join(fold_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(fold_dir, "test.csv"))
    
    # 分离特征和标签
    target_col = 'Class'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"训练集: {X_train.shape}，测试集: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="模型"):
    """训练和评估模型"""
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # 打印结果
    print(f"\n{model_name} 结果:")
    print(f"- 准确率: {acc:.4f}")
    print(f"- 宏平均F1分数: {f1:.4f}")
    
    return acc, f1, y_pred

def main():
    print("开始测试简单模型...")
    
    # 1. 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 2. 尝试决策树模型
    print("\n" + "="*50)
    print("1. 决策树模型测试")
    print("="*50)
    
    try:
        # 创建简单决策树
        dt_model = DecisionTreeClassifier(
            max_depth=5, 
            random_state=42
        )
        
        # 评估决策树
        dt_acc, dt_f1, _ = evaluate_model(
            dt_model, X_train, y_train, X_test, y_test, 
            model_name="决策树 (max_depth=5)"
        )
        
    except Exception as e:
        print(f"决策树模型出错: {e}")
        print("尝试更简单的基准模型...")
    
    # 3. 使用虚拟模型作为基准
    print("\n" + "="*50)
    print("2. 基准模型测试")
    print("="*50)
    
    # 创建虚拟模型（最频繁类别）
    dummy_model = DummyClassifier(
        strategy='most_frequent',
        random_state=42
    )
    
    # 评估虚拟模型
    dummy_acc, dummy_f1, _ = evaluate_model(
        dummy_model, X_train, y_train, X_test, y_test,
        model_name="虚拟分类器 (最频繁类别)"
    )
    
    # 4. 结果对比
    print("\n" + "="*50)
    print("3. 结果对比")
    print("="*50)
    
    try:
        improvement = dt_acc - dummy_acc
        print(f"决策树相对于基准的提升:")
        print(f"- 准确率提升: {improvement:.4f}")
        print(f"- 相对提升: {improvement/dummy_acc*100:.1f}%" if dummy_acc > 0 else "- 基准为0")
    except:
        print("无法对比结果，决策树可能未成功运行")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()