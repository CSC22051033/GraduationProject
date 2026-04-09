from carClaims import split
from DeepLearning_CNN import CNNModel
from DeepLearning_RNN import RNNModel
from DeepLearning_Mixed import ImprovedCNNLSTM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_model(model, X, y, model_name):
    y_pred = []
    for i in range(len(X)):
        _, pred = model.predict(X.iloc[i])
        y_pred.append(pred)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
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
                ):
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    cnn = CNNModel()
    rnn = RNNModel()
    mix = ImprovedCNNLSTM()

    if fit_flag:
        cnn.fit(X_train, y_train, X_valid, y_valid,50)
        rnn.fit(X_train, y_train, X_valid, y_valid,50)
        mix.fit(X_train, y_train, X_valid, y_valid,50)

    # 定义模型列表（名称和实例）
    models = [
        ('CNNModel', CNNModel()),
        ('RNNModel', RNNModel()),
        ('ImprovedCNNLSTM', ImprovedCNNLSTM())
    ]

    # 用于存储所有模型的评估结果
    results = []

    for name, model in models:
        print(f"\n--- Evaluating {name} ---")
        model.load()  # 加载预训练模型
        
        model._plot_confusion_matrix_minimal(X_test, y_test, 0.5)

        print("预测（前10个数据）:\n")
        for i in range(10):
            model.predict(X_test.iloc[i], y_test.iloc[i])

        # 对整个测试集进行评估
        metrics = evaluate_model(model, X_test, y_test, name)
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
    X_test.to_csv('output/X_test.csv', index=False)
    y_test.to_csv('output/y_test.csv')
    print(X_train.head())
    deep_learning(X_train, y_train, X_valid, y_valid, X_test, y_test)