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

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    print(X_train.head())

    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    res_file = 'output/res'  # 结果文件路径

    cnn = CNNModel()
    rnn = RNNModel()
    mix = ImprovedCNNLSTM()
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

        print("Sample predictions (first 10 test samples):")
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
