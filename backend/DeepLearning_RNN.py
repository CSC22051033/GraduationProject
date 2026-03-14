from carClaims import BaseModel, EarlyStopping

# ====== PyTorch深度学习框架 ======
import torch                    # PyTorch深度学习框架，提供张量计算和自动求导功能
import torch.nn as nn           # PyTorch神经网络模块，提供网络层、损失函数等
import torch.optim as optim

# ====== 数据处理相关库 ======
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算

# ====== 模型评估指标 ======
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score   # 分类模型评估指标
# roc_auc_score: 接收者操作特征曲线下面积，衡量分类器性能
# accuracy_score: 准确率，正确分类样本占总样本比例
# precision_score: 精确率，预测为正的样本中实际为正的比例
# recall_score: 召回率，实际为正的样本中被正确预测为正的比例

# ====== 可视化 ======
from tqdm import tqdm
import matplotlib.pyplot as plt     # 绘图库，用于数据分布、结果曲线等可视化
# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

class ImprovedLSTM(nn.Module):
    def __init__(self, n_features: int):
        super(ImprovedLSTM, self).__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=128, num_layers=2,
                             batch_first=True, dropout=0.3, bidirectional=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B,1,F)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]   # (B,128)
        last_hidden = self.bn1(last_hidden)
        last_hidden = self.relu(self.fc1(last_hidden))
        last_hidden = self.bn2(last_hidden)
        last_hidden = self.dropout(last_hidden)
        return self.fc2(last_hidden)

class RNNModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.pt_path = 'output/model/best_lstm.pt'
        self.model_path = 'output/model/rnn_model.pth'
        self.fig_hist_path = 'output/img/rnn_training_history.png'
        self.fig_conf_path = 'output/img/rnn_confusion_matrix.png'
        self.fig_opti_path = 'output/img/rnn_threshold_optimization.png'
        self.model_class = ImprovedLSTM

    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50):
        torch.manual_seed(42)
        self.n_features = X_train.shape[1]
        self.pos_weight = self.calc_pos_weight(y_train)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))

        train_loader = self.df_to_loader(X_train, y_train, shuffle=True)
        valid_loader = self.df_to_loader(X_valid, y_valid, shuffle=False)

        self.model = self.model_class(self.n_features).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10//2)
        early_stop = EarlyStopping(patience=10, verbose=True, save_path=self.pt_path)

        history = {'train_loss': [], 'val_loss': [],
                   'train_auc': [], 'val_auc': [],
                   'train_acc': [], 'val_acc': [],
                   'train_pre': [], 'val_pre': [],
                   'train_rec': [], 'val_rec': []}

        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            train_loss, train_preds, train_labels = 0.0, [], []
            for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} - train', disable=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_preds.append(torch.sigmoid(out).detach().cpu().numpy())
                train_labels.append(yb.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_preds_np = np.concatenate(train_preds)
            train_labels_np = np.concatenate(train_labels)
            train_auc = roc_auc_score(train_labels_np, train_preds_np)
            train_preds_binary = (train_preds_np > 0.5).astype(int)
            if len(np.unique(train_labels_np)) > 1:
                train_acc = accuracy_score(train_labels_np, train_preds_binary)
                train_pre = precision_score(train_labels_np, train_preds_binary, zero_division=0)
                train_rec = recall_score(train_labels_np, train_preds_binary, zero_division=0)
            else:
                train_acc = train_pre = train_rec = 0.0

            # 验证
            val_loss, val_auc, val_acc, val_pre, val_rec = self._evaluate(valid_loader)
            scheduler.step(val_auc)

            # 记录
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            history['train_acc'].append(train_acc)
            history['train_pre'].append(train_pre)
            history['train_rec'].append(train_rec)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_acc'].append(val_acc)
            history['val_pre'].append(val_pre)
            history['val_rec'].append(val_rec)

            print(f'Epoch {epoch:03d} | '
                  f'Train Loss {train_loss:.4f} | Train AUC {train_auc:.4f} | '
                  f'Val Loss {val_loss:.4f} | Val AUC {val_auc:.4f} | '
                  f'LR {optimizer.param_groups[0]["lr"]:.2e}')

            if early_stop(val_auc, self.model):
                print(f'Early stop at epoch {epoch}')
                break

        self.model.load_state_dict(torch.load(self.pt_path, map_location=self.device))
        self.save()
        self._plot_metrics(history)
        return history
    