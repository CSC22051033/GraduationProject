import torch
import torch.nn as nn

# ====== 文件操作 ======
import os     # 提供文件路径操作功能，用于创建目录和文件管理

# ====== 模型评估指标 ======
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score   # 分类模型评估指标

# ====== 数据处理相关库 ======
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算

# ====== 可视化 ======
from tqdm import tqdm

from carClaims import split     # 数据集划分 - 分割原始数据集为训练集、验证集和测试集
from carClaims import EarlyStopping
from DeepLearning_RNN import RNNModel

class CNN_LSTM(nn.Module):
    """
        1D-CNN + LSTM 用于表格特征二分类
    """
    def __init__(self, n_features: int):
        super().__init__()

        hidden_size = 64
        num_layers = 2

        # 1. 1D-CNN 做局部特征提取
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)

        # 2. LSTM 层
        # CNN 输出通道 64 作为 LSTM 的 input_size
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.)

        # 3. 分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, F)
        x = x.unsqueeze(1)                # (B, 1, F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))   # (B, 64, F)

        x = x.transpose(1, 2)             # (B, F, 64)
        out, (h_n, _) = self.lstm(x)      # h_n: (num_layers, B, hidden)
        x = h_n[-1]                       # (B, hidden)

        logit = self.fc(x)                # (B, 1)  不要 squeeze
        return logit                      
    
class ImprovedCNNLSTM(RNNModel):
    def __init__(self):
        super().__init__()
        self.pt_path = 'output/model/best_mix.pt'
        self.model_path = 'output/model/mix_model.pth'
        self.fig_hist_path = 'output/img/mix_training_history.png'
        self.fig_conf_path = 'output/img/mix_confusion_matrix.png'
        self.fig_opti_path = 'output/img/mix_threshold_optimization.png'

    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50):
        # 1. 基础设置
        torch.manual_seed(42)
        self.n_features = X_train.shape[1]
        self.pos_weight = self.calc_pos_weight(y_train)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        verbose = True
        
        # 2. 构造 DataLoader
        train_loader = self.df_to_loader(X_train, y_train, shuffle=True)
        valid_loader = self.df_to_loader(X_valid, y_valid, shuffle=False)

        # 3. 模型、优化器、早停
        self.model = CNN_LSTM(self.n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=7//2)
        early_stop = EarlyStopping(patience=7, verbose=verbose, save_path=self.pt_path)

        # 初始化历史记录
        history = {'train_loss': [], 'val_loss': [],
                   'train_auc': [], 'val_auc': [],
                   'train_acc': [], 'val_acc': [],
                   'train_pre': [], 'val_pre': [],
                   'train_rec': [], 'val_rec': []}
        
        # 4. 训练循环
        for epoch in range(1, epochs + 1):
            # ---------- 训练 ----------
            self.model.train()
            train_loss, train_preds, train_labels = 0.0, [], []
            
            for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} - train', disable=not verbose):
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
            
            # 计算训练集的其他指标
            train_preds_binary = (train_preds_np > 0.5).astype(int)
            if len(np.unique(train_labels_np)) > 1:
                train_acc = accuracy_score(train_labels_np, train_preds_binary)
                train_pre = precision_score(train_labels_np, train_preds_binary, zero_division=0)
                train_rec = recall_score(train_labels_np, train_preds_binary, zero_division=0)
            else:
                train_acc = train_pre = train_rec = 0.0

            # ---------- 验证 ----------
            val_loss, val_auc, val_acc, val_pre, val_rec = self._evaluate(valid_loader)
            scheduler.step(val_auc)

            # ---------- 记录历史 ----------
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

            # ---------- 日志 ----------
            if verbose:
                print(f'Epoch {epoch:03d} | '
                      f'Train Loss {train_loss:.4f} | Train AUC {train_auc:.4f} | '
                      f'Val Loss {val_loss:.4f} | Val AUC {val_auc:.4f} | '
                      f'LR {optimizer.param_groups[0]["lr"]:.2e}')

            # ---------- 早停 ----------
            if early_stop(val_auc, self.model):
                if verbose:
                    print(f'Early stop at epoch {epoch}')
                break

        # 5. 训练结束后加载最优权重
        self.model.load_state_dict(torch.load(self.pt_path, map_location=self.device))
        if verbose:
            print('Loaded best model.')

        if verbose:
            print('Loaded best model.')
        
        # 训练结束后自动保存完整模型
        self.save()

        self._plot_metrics(history)  # 绘制训练历史
        return history
    
    def load(self):
        """
        加载 CNN-LSTM 完整模型
        """
        filepath = self.model_path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # 重建 CNN-LSTM
        self.n_features = checkpoint['n_features']
        self.model = CNN_LSTM(self.n_features).to(self.device)   # 用你的类名
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 重建损失函数
        if checkpoint.get('pos_weight') is not None:
            self.pos_weight = checkpoint['pos_weight'].to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        print(f'CNN-LSTM model loaded from {filepath}')
        return self

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    model = ImprovedCNNLSTM()
    # model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    model.load()
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)