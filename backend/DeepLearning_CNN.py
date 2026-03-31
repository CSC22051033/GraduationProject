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
    def __init__(self, n_features: int, dropout_rate=0.3, weight_decay=1e-3):
        super().__init__()
        # 降低卷积核数量（原 32->64，改为 16->32）
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout1d(dropout_rate)   # 卷积层后添加 Dropout1d
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout1d(dropout_rate)
        
        # 保留池化层
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        
        # 降低全连接层神经元数量（原 128->64->1，改为 64->32->1）
        self.fc1 = nn.Linear(64, 32)   # 32 (avg) + 32 (max) = 64
        self.dp = nn.Dropout(0.5)      # 全连接层 dropout 保持或略增
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B,1,F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        avg = self.pool_avg(x).squeeze(-1)     # (B,32)
        maxp = self.pool_max(x).squeeze(-1)    # (B,32)
        x = torch.cat([avg, maxp], dim=1)      # (B,64)
        x = torch.relu(self.fc1(x))
        x = self.dp(x)
        return self.fc2(x)

class CNNModel(BaseModel):
    def __init__(self, lr=5e-4, device=None, weight_decay=1e-2, pos_weight=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.weight_decay = weight_decay
        # None：在 fit 内按 y_train 计算 neg/pos（与 RNN 一致，与 SMOTE 后的训练分布一致）
        # 传入数值则强制使用该 pos_weight（不做原始分布推算）
        self.pos_weight = pos_weight
        self.model_class = CNN1D
        # 路径覆盖
        self.pt_path = 'output/model/best_cnn.pt'
        self.model_path = 'output/model/cnn_trained.pth'
        self.fig_hist_path = 'output/img/cnn_training_history.png'
        self.fig_conf_path = 'output/img/cnn_confusion_matrix.png'

    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, patience=10,
            lr_scheduler_factor=0.5, lr_scheduler_patience=3):
        os.makedirs('output/model', exist_ok=True)
        os.makedirs('output/img', exist_ok=True)

        self.n_features = X_train.shape[1]
        self.model = self.model_class(self.n_features).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )

        if self.pos_weight is None:
            self.pos_weight = self.calc_pos_weight(y_train).to(self.device)
        else:
            pw = self.pos_weight
            if not isinstance(pw, torch.Tensor):
                pw = torch.tensor(float(pw), dtype=torch.float32, device=self.device)
            else:
                pw = pw.to(self.device)
            self.pos_weight = pw
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

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

            scheduler.step(val_auc)

            if early_stopping(val_auc, self.model):
                print(f'Early stopped at epoch {epoch}')
                break

        # 加载最优模型并保存
        self.model.load_state_dict(torch.load(self.pt_path))
        self.save()
        self._plot_metrics(history)
        print('Finished, best AUC:', early_stopping.best_auc)  
