# ====== 数据处理相关库 ======
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算

# ====== 文件操作 ======
import os     # 提供文件路径操作功能，用于创建目录和文件管理

# ====== PyTorch深度学习框架 ======
import torch                      # PyTorch深度学习框架，提供张量计算和自动求导功能

import torch.nn as nn             # PyTorch神经网络模块，提供网络层、损失函数等
import torch.optim as optim       # PyTorch优化器模块，提供梯度下降等优化算法

# ====== 模型评估指标 ======
from sklearn.metrics import accuracy_score, precision_score, recall_score   # 分类模型评估指标
# accuracy_score: 准确率，正确分类样本占总样本比例
# precision_score: 精确率，预测为正的样本中实际为正的比例
# recall_score: 召回率，实际为正的样本中被正确预测为正的比例

# ====== 可视化 ======
from tqdm import tqdm
import matplotlib.pyplot as plt     # 绘图库，用于数据分布、结果曲线等可视化
# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

from carClaims import BaseModel, EarlyStopping

class CNN1D(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 64)
        self.dp = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B,1,F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # (B,64)
        x = torch.relu(self.fc1(x))
        x = self.dp(x)
        return self.fc2(x)

class CNNModel(BaseModel):
    def __init__(self, lr=1e-3, device=None, pos_weight=1.0):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.pos_weight = pos_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        self.model_class = CNN1D
        # 路径覆盖
        self.pt_path = 'output/model/best_cnn.pt'
        self.model_path = 'output/model/cnn_trained.pth'
        self.fig_hist_path = 'output/img/cnn_training_history.png'
        self.fig_conf_path = 'output/img/cnn_confusion_matrix.png'
        self.fig_opti_path = 'output/img/cnn_threshold_optimization.png'

    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, patience=10):
        os.makedirs('output/model', exist_ok=True)
        os.makedirs('output/img', exist_ok=True)

        self.n_features = X_train.shape[1]
        self.model = self.model_class(self.n_features).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 根据训练集重新计算 pos_weight
        pos_weight_tensor = self.calc_pos_weight(y_train).to(self.device)
        self.pos_weight = pos_weight_tensor     # 更新属性为 Tensor
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        train_loader = self.df_to_loader(X_train, y_train, shuffle=True)
        valid_loader = self.df_to_loader(X_valid, y_valid, shuffle=False)

        history = {k: [] for k in
                   ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                    'train_pre', 'val_pre', 'train_rec', 'val_rec']}

        early_stopping = EarlyStopping(patience=patience, save_path=self.pt_path)

        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch:02d} Training')
            for xb, yb in train_loader_tqdm:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_preds.append(torch.sigmoid(out).detach().cpu().numpy())
                train_labels.append(yb.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_preds_np = np.concatenate(train_preds)
            train_labels_np = np.concatenate(train_labels)
            train_preds_binary = (train_preds_np > 0.5).astype(int)
            train_acc = accuracy_score(train_labels_np, train_preds_binary)
            train_pre = precision_score(train_labels_np, train_preds_binary, zero_division=0)
            train_rec = recall_score(train_labels_np, train_preds_binary, zero_division=0)

            # 验证
            val_loss, val_auc, val_acc, val_pre, val_rec = self._evaluate(valid_loader)

            # 记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_pre'].append(train_pre)
            history['train_rec'].append(train_rec)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_pre'].append(val_pre)
            history['val_rec'].append(val_rec)

            print(f'Epoch {epoch:02d} | '
                  f'train loss {train_loss:.4f} acc {train_acc:.4f} pre {train_pre:.4f} rec {train_rec:.4f} | '
                  f'val loss {val_loss:.4f} acc {val_acc:.4f} pre {val_pre:.4f} rec {val_rec:.4f} | '
                  f'val AUC {val_auc:.4f}')

            if early_stopping(val_auc, self.model):
                print(f'Early stopped at epoch {epoch}')
                break

        # 加载最优模型并保存
        self.model.load_state_dict(torch.load(self.pt_path))
        self.save()
        self._plot_metrics(history)
        print('Finished, best AUC:', early_stopping.best_auc)  
