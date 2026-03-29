from carClaims import split
from DeepLearning_CNN import CNNModel
from DeepLearning_RNN import RNNModel
from DeepLearning_Mixed import ImprovedCNNLSTM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_model(model, X, y, model_name, pred_threshold=0.5):
    y_pred = model.predict_labels(X, pred_threshold=pred_threshold)
    y_true = y.values if hasattr(y, 'values') else y
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def deep_learning(X_train, y_train, X_valid, y_valid, X_test, y_test,   # 数据
                  res_file = 'output/res',      # 输出结果路径
                  fit_flag = True,              # 是否进行模型训练
                  pred_threshold = 0.5,         # 概率超过该值判为欺诈；略提高可提升精确率、降低召回
                ):
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    cnn = CNNModel()
    # rnn = RNNModel()
    # mix = ImprovedCNNLSTM()

    if fit_flag:
        cnn.fit(X_train, y_train, X_valid, y_valid,50)
        # rnn.fit(X_train, y_train, X_valid, y_valid,50)
        # mix.fit(X_train, y_train, X_valid, y_valid,50)

    # 定义模型列表（名称和实例）
    models = [
        ('CNNModel', CNNModel()),
        # ('RNNModel', RNNModel()),
        # ('ImprovedCNNLSTM', ImprovedCNNLSTM())
    ]

    # 用于存储所有模型的评估结果
    results = []

    print(f"测试集评估 pred_threshold = {pred_threshold}")

    for name, model in models:
        print(f"\n--- Evaluating {name} ---")
        model.load()  # 加载预训练模型
        
        model._plot_confusion_matrix_minimal(X_test, y_test, pred_threshold=pred_threshold)

        print("预测（前10个数据）:\n")
        for i in range(10):
            model.predict(X_test.iloc[i], y_test.iloc[i], best_threshold=pred_threshold)

        # 对整个测试集进行评估
        metrics = evaluate_model(model, X_test, y_test, name, pred_threshold=pred_threshold)
        results.append(metrics)
        print(f"{name} overall metrics: {metrics}")

    # 将结果写入文件
    with open(res_file, 'w') as f:
        f.write("model,accuracy,precision,recall,f1\n")
        for res in results:
            f.write(f"{res['model']},{res['accuracy']:.4f},{res['precision']:.4f},{res['recall']:.4f},{res['f1']:.4f}\n")
    
    print(f"\nEvaluation results saved to {res_file}")

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound', True)
    print("训练集\n")
    print(X_train.head())
    print("验证集\n")
    print(X_valid.head())
    print("测试集\n")
    print(X_test.head())
    deep_learning(X_train, y_train, X_valid, y_valid, X_test, y_test)
    