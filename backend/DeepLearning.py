from carClaims import split
from DeepLearning_CNN import CNNModel
from DeepLearning_RNN import RNNModel
from DeepLearning_Mixed import ImprovedCNNLSTM


if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    print(X_train.head())

    cnn = CNNModel()
    cnn.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    # cnn.load()
    for i in range(100):
        cnn.predict(X_test.iloc[i], y_test.iloc[i])
   
    model = RNNModel()
    model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    # model.load()
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])

    model = ImprovedCNNLSTM()
    model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    # model.load()
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
