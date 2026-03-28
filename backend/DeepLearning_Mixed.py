import torch
import torch.nn as nn

from carClaims import split     # 数据集划分 - 分割原始数据集为训练集、验证集和测试集
from DeepLearning_RNN import RNNModel

class CNN_LSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B,1,F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))   # (B,64,F)
        x = x.transpose(1, 2)                    # (B,F,64)
        out, (h_n, _) = self.lstm(x)             # h_n: (num_layers,B,64)
        x = h_n[-1]                               # (B,64)
        return self.fc(x)

class ImprovedCNNLSTM(RNNModel):
    def __init__(self):
        super().__init__()
        # 覆盖路径与模型类
        self.pt_path = 'output/model/best_mix.pt'
        self.model_path = 'output/model/mix_model.pth'
        self.fig_hist_path = 'output/img/mix_training_history.png'
        self.fig_conf_path = 'output/img/mix_confusion_matrix.png'
        self.model_class = CNN_LSTM
        
if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    model = ImprovedCNNLSTM()
    # model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    model.load()
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)