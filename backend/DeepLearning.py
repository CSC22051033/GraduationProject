from carClaims import split
from DeepLearning_CNN import CNNModel
from DeepLearning_RNN import RNNModel
from DeepLearning_Mixed import ImprovedCNNLSTM
import torch   

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    print(X_train.head())

    cnn = CNNModel()
    cnn.fit_with_visual(X_train, y_train, X_valid, y_valid, epochs=50)
    # 测试集表现
    cnn.model.load_state_dict(torch.load('output/model/best_cnn.pt'))
    # cnn.load_model('output/model/cnn_trained.pth')
    best_threshold = cnn.optimize_threshold(X_valid, y_valid)
    cnn._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)

    model = RNNModel()
    model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    # model.load()
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)

    model = ImprovedCNNLSTM()
    model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    # model.load()
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)