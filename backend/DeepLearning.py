from carClaims import split
from DeepLearning_CNN import CNNModel
from DeepLearning_RNN import RNNModel
from DeepLearning_Mixed import ImprovedCNNLSTM


if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    print(X_train.head())

    cnn = CNNModel()
    cnn.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    cnn.load()
    best_threshold = cnn.optimize_threshold(X_valid, y_valid)
    for i in range(100):
        cnn.predict(X_test.iloc[i], y_test.iloc[i])
    cnn._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)
   
"""    model = RNNModel()
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
"""