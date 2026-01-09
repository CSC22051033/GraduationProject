# ====== 数据预处理 ======
from carClaims import split     # 数据集划分 - 分割原始数据集为训练集、验证集和测试集
from carClaims import Model, EarlyStopping

# ====== 文件操作 ======
import os     # 提供文件路径操作功能，用于创建目录和文件管理

# ====== PyTorch深度学习框架 ======
import torch                    # PyTorch深度学习框架，提供张量计算和自动求导功能
import torch.nn as nn           # PyTorch神经网络模块，提供网络层、损失函数等

# ====== 数据处理相关库 ======
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算

# ====== 模型评估指标 ======
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score   # 分类模型评估指标
# roc_auc_score: 接收者操作特征曲线下面积，衡量分类器性能
# accuracy_score: 准确率，正确分类样本占总样本比例
# precision_score: 精确率，预测为正的样本中实际为正的比例
# recall_score: 召回率，实际为正的样本中被正确预测为正的比例

from tqdm import tqdm

# ====== 可视化 ======
import matplotlib.pyplot as plt     # 绘图库，用于数据分布、结果曲线等可视化
# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

class ImprovedLSTM(nn.Module):
    def __init__(self, n_features: int):
        super(ImprovedLSTM, self).__init__()
        self.n_features = n_features

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3 if 2 > 1 else 0,
            bidirectional=False
        )
        # 双向LSTM的因子
        lstm_factor = 2
        self.bn1 = nn.BatchNorm1d(128 * lstm_factor)     # 批归一化
        # 全连接层
        self.fc1 = nn.Linear(128 * lstm_factor, 64)
        self.bn2 = nn.BatchNorm1d(64)
        # Dropout
        self.dropout = nn.Dropout(0.3)
        # 输出层
        self.fc2 = nn.Linear(64, 1)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x形状: (batch_size, sequence_length, n_features)
        或 (batch_size, n_features) 如果是单时间步
        """
        # 如果输入是2D，添加时间步维度
        if x.dim() == 2:
            # (batch_size, n_features) -> (batch_size, 1, n_features)
            x = x.unsqueeze(1)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # 如果是双向LSTM，需要拼接最后两个方向
        if self.lstm.bidirectional:
            # 取前向和后向的最后一个时间步并拼接
            forward_last = lstm_out[:, -1, :self.lstm.hidden_size]
            backward_last = lstm_out[:, 0, self.lstm.hidden_size:]
            last_hidden = torch.cat((forward_last, backward_last), dim=1)
        else:
            last_hidden = lstm_out[:, -1, :]
        
        # 批归一化和全连接层
        last_hidden = self.bn1(last_hidden)
        last_hidden = self.relu(self.fc1(last_hidden))
        last_hidden = self.bn2(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        output = self.fc2(last_hidden)
        return output

class RNNModel(Model):
    def __init__(self):
        super().__init__()

    def _plot_metrics(self, hist):
        """修复的绘图函数 - 确保所有需要的键都存在"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 修改：使用中文小标题
        metrics = [('训练和验证损失', 'train_loss', 'val_loss'),
                ('训练和验证准确率', 'train_acc', 'val_acc'),
                ('训练和验证精确率', 'train_pre', 'val_pre'),
                ('训练和验证召回率', 'train_rec', 'val_rec')]
        
        # 修改：中文图例
        label_map = {'train': '训练', 'val': '验证'}
        
        # 检查并绘制存在的指标
        plot_count = 0
        for ax, (title, tr_key, val_key) in zip(axes, metrics):
            # 检查键是否存在且有数据
            if tr_key in hist and len(hist[tr_key]) > 0 and val_key in hist and len(hist[val_key]) > 0:
                ax.plot(hist[tr_key], label=label_map['train'], 
                    linewidth=2, marker='o', markersize=4, markevery=1)
                ax.plot(hist[val_key], label=label_map['val'], 
                    linewidth=2, marker='o', markersize=4, markevery=1)
                
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('Epoch', fontsize=12)
                
                # 修改：y轴标签根据指标类型设置
                if 'loss' in tr_key:
                    ax.set_ylabel('Loss', fontsize=12)
                elif 'acc' in tr_key:
                    ax.set_ylabel('Accuracy', fontsize=12)
                elif 'pre' in tr_key:
                    ax.set_ylabel('Precision', fontsize=12)
                elif 'rec' in tr_key:
                    ax.set_ylabel('Recall', fontsize=12)
                    
                ax.legend(fontsize=11, loc='best')
                ax.grid(alpha=0.3) 
                ax.tick_params(labelsize=10)
                
                plot_count += 1
            else:
                ax.remove()  # 移除没有数据的子图
        
        # 如果所有指标都缺失，显示提示
        if plot_count == 0:
            fig.clear()
            plt.close(fig)
            print("警告: 没有可绘制的训练指标数据")
            return
        
        # 调整布局，移除多余的空子图
        for i in range(plot_count, len(axes)):
            if i < len(axes):
                axes[i].remove()
        
        plt.tight_layout()
        plt.savefig('output/img/rnn_training_history.png', dpi=150)
        plt.close(fig)

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
        self.model = ImprovedLSTM(self.n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=7//2)
        early_stop = EarlyStopping(patience=7, verbose=verbose, save_path='output/model/best_lstm.pt')

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
        self.model.load_state_dict(torch.load('output/model/best_lstm.pt', map_location=self.device))
        if verbose:
            print('Loaded best model.')

        if verbose:
            print('Loaded best model.')
        
        # 新增：训练结束后自动保存完整模型
        self.save('output/model/rnn_model.pth')
        if verbose:
            print('Saved full model to output/model/rnn_model.pth')

        self._plot_metrics(history)  # 绘制训练历史
        return history

    # -------------------- 验证集评估 --------------------
    def _evaluate(self, data_loader):
        """修复的评估函数 - 返回所有需要的指标"""
        self.model.eval()
        total_loss, preds, labels = 0.0, [], []
        
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                preds.append(torch.sigmoid(out).cpu().numpy())
                labels.append(yb.cpu().numpy())
        
        total_loss /= len(data_loader.dataset)
        preds_np = np.concatenate(preds)
        labels_np = np.concatenate(labels)
        
        # 计算AUC
        if len(np.unique(labels_np)) > 1:
            auc = roc_auc_score(labels_np, preds_np)
        else:
            auc = 0.5  # 默认值，如果只有一类
        
        # 计算其他指标
        preds_binary = (preds_np > 0.5).astype(int)
        if len(np.unique(labels_np)) > 1:
            acc = accuracy_score(labels_np, preds_binary)
            pre = precision_score(labels_np, preds_binary, zero_division=0)
            rec = recall_score(labels_np, preds_binary, zero_division=0)
        else:
            acc = pre = rec = 0.0
        
        return total_loss, auc, acc, pre, rec

    def predict(self, x_single, y_true=None, best_threshold=0.5):
        # 1. 统一转 numpy 并确保是 1-D
        if hasattr(x_single, 'values'):        # pd.Series
            x_single = x_single.values
        x_single = np.asarray(x_single, dtype=np.float32).squeeze()
        if x_single.ndim != 1:
            raise ValueError('x_single 必须是 1 维向量')

        # 2. 构造 batch=1 的 DataLoader（复用已有函数）
        X = x_single.reshape(1, -1)            # (1, n_features)
        loader = self.df_to_loader(X, shuffle=False)  # y 占位即可

        # 3. 预测
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logit = self.model(xb)
                prob  = torch.sigmoid(logit).cpu().item()

        pred_label = int(prob > best_threshold)

        # 4. 打印对比
        if y_true is not None:
            print(f'真实标签: {int(y_true)}  |  预测概率: {prob:.4f}  |  预测类别: {pred_label}')
        return prob, pred_label

    def save(self, filepath='output/model/rnn_model.pth'):
        """
        保存完整模型
        Args:
            filepath: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型状态和必要参数
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'n_features': self.n_features,
            'pos_weight': self.pos_weight.cpu() if hasattr(self, 'pos_weight') else None
        }
        torch.save(save_dict, filepath)
        print(f'Model saved to {filepath}')
    
    def load(self, filepath='output/model/rnn_model.pth'):
        """
        加载完整模型
        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # 加载保存的数据
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 重建模型
        self.n_features = checkpoint['n_features']
        self.model = ImprovedLSTM(self.n_features).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载权重参数
        if checkpoint['pos_weight'] is not None:
            self.pos_weight = checkpoint['pos_weight'].to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        print(f'Model loaded from {filepath}')
        return self

    def _plot_confusion_matrix_minimal(self, X_test, y_test, best_threshold):
        """最小化混淆矩阵绘制"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        
        # 收集预测结果
        y_pred = []
        for i in range(min(2500, len(X_test))):  # 限制数量避免太长
            _, pred = self.predict(x_single = X_test.iloc[i], best_threshold=best_threshold)
            y_pred.append(pred)
        
        # 截取相同长度的真实标签
        y_true = y_test.iloc[:len(y_pred)]
        
        # 计算并绘制混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        
        # 显示数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), 
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        plt.tight_layout()
        # plt.show()
        plt.savefig('output/img/rnn_confusion_matrix.png', dpi=150)
        plt.close()

    def optimize_threshold(self, X_valid, y_valid, step=0.01):
        """
        简洁的阈值优化函数
        使用验证集优化阈值，可视化最佳阈值和F1分数曲线
        返回最佳阈值
        
        参数:
            X_valid: 验证集特征
            y_valid: 验证集标签 (0/1)
            step: 阈值搜索步长，默认0.01
        
        返回:
            best_threshold: 最佳阈值
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import f1_score
        
        # 获取验证集预测概率
        self.model.eval()
        valid_loader = self.df_to_loader(X_valid, y_valid, shuffle=False)
        
        y_true_list = []
        y_score_list = []
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_score_list.append(probs)
                y_true_list.append(yb.cpu().numpy())
        
        y_true = np.concatenate(y_true_list).flatten()
        y_scores = np.concatenate(y_score_list).flatten()
        
        # 在不同阈值下计算F1分数
        thresholds = np.arange(0, 1 + step, step)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        # 找到最佳阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # 可视化
        plt.figure(figsize=(6, 6))
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
        plt.axvline(x=best_threshold, color='red', linestyle='--', 
                    label=f'最佳阈值: {best_threshold:.3f}')
        plt.plot(best_threshold, best_f1, 'ro', markersize=10)
        
        plt.xlabel('阈值')
        plt.ylabel('F1分数')
        plt.title(f'阈值优化结果 (最佳阈值: {best_threshold:.3f}, F1: {best_f1:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        os.makedirs('output/img', exist_ok=True)
        plt.savefig('output/img/rnn_threshold_optimization.png', dpi=150, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # 展示不同阈值下的性能指标
        print("\n阈值优化结果:")
        print("-" * 60)
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"最佳F1分数: {best_f1:.4f}")
        print("-" * 60)
        
        # 展示几个关键阈值点的性能
        key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print("\n不同阈值下的F1分数:")
        for th in key_thresholds:
            idx = int(th / step)
            if idx < len(f1_scores):
                print(f"  阈值 {th:.1f}: F1 = {f1_scores[idx]:.4f}")
        
        return best_threshold

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    model = RNNModel()
    # model.fit(X_train, y_train, X_valid, y_valid, epochs=50)
    model.load()
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    for i in range(10):
        model.predict(X_test.iloc[i], y_test.iloc[i])
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)


    