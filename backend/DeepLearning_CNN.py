# ====== 数据预处理 ======
from carClaims import split     # 数据集划分 - 分割原始数据集为训练集、验证集和测试集

# ====== 数据处理相关库 ======
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算

# ====== 文件操作 ======
import os     # 提供文件路径操作功能，用于创建目录和文件管理

# ====== PyTorch深度学习框架 ======
import torch                      # PyTorch深度学习框架，提供张量计算和自动求导功能
from torch.utils.data import TensorDataset, DataLoader    # 数据集封装和数据加载工具

import torch.nn as nn             # PyTorch神经网络模块，提供网络层、损失函数等
import torch.optim as optim       # PyTorch优化器模块，提供梯度下降等优化算法

# ====== 模型评估指标 ======
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score   # 分类模型评估指标
# roc_auc_score: 接收者操作特征曲线下面积，衡量分类器性能
# accuracy_score: 准确率，正确分类样本占总样本比例
# precision_score: 精确率，预测为正的样本中实际为正的比例
# recall_score: 召回率，实际为正的样本中被正确预测为正的比例

# ====== 可视化 ======
import matplotlib.pyplot as plt     # 绘图库，用于数据分布、结果曲线等可视化
# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# ========== 早停工具类 ==========
class EarlyStopping:
    """
    基于验证 AUC 的早停策略
    Args:
        patience (int): AUC 连续 patience 轮没有提升就停
        verbose (bool): 是否打印早停信息
        save_path (str): 最优模型保存路径
    """
    def __init__(self, patience=7, verbose=True, save_path='output/model/best_cnn.pt'):
        self.patience = patience
        self.verbose  = verbose
        self.save_path = save_path
        self.best_auc   = 0.0
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_auc, model):
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f'EarlyStopping: New best AUC = {val_auc:.4f}, model saved.')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: patience = {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('EarlyStopping: Triggered!')
        return self.early_stop

class CNN1D(nn.Module):
    """
        一维 CNN 用于表格特征二分类
    """
    def __init__(self, n_features: int):
        super().__init__()
        # 卷积层: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)   # -> (B,32,F)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # -> (B,64,F)
        self.bn2   = nn.BatchNorm1d(64)

        # 全局最大池化后变成 (B,64)
        self.fc1 = nn.Linear(64, 64)
        self.dp  = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B,F) -> (B,1,F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # (B,64)
        x = torch.relu(self.fc1(x))
        x = self.dp(x)
        return self.fc2(x)

class CNNModel:
    def __init__(self, lr: float = 1e-3, device: str = None, pos_weight: float = 1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = None
        self.lr     = lr
        self.optimizer = None
        self.pos_weight = pos_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.n_features = None  # 记录输入特征维度

    def df_to_loader(self, X, y=None, shuffle=True):
        # ---- 1. 统一把 X 转成 numpy ----
        if hasattr(X, 'values'):          # pandas 对象
            X = X.values
        X_tensor = torch.tensor(X.astype(np.float32))

        # ---- 2. 处理 y ----
        if y is None:                     # 预测时可以不传 y
            y_tensor = torch.zeros(len(X_tensor))  # 占位，后面不会用
        else:
            if hasattr(y, 'values'):      # pandas 对象
                y = y.values
            y = y.squeeze() if y.ndim > 1 else y
            y_tensor = torch.tensor(y.astype(np.float32)).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader  = DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=0)
        return loader

    @staticmethod
    def calc_pos_weight(y):
        """输入 0/1 向量，返回负/正比例 tensor"""
        y = np.asarray(y).squeeze()
        pos = y.sum()
        neg = len(y) - pos
        return torch.tensor(neg / max(pos, 1))   # 避免除 0

    def save_model(self, filepath='output/model/cnn_model.pth'):
            """
            保存模型参数和必要的训练信息
            
            参数:
                filepath: 模型保存路径
            """
            if self.model is None:
                raise ValueError("模型未训练，无法保存")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型参数和训练信息
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'n_features': self.n_features,
                'lr': self.lr,
                'pos_weight': self.pos_weight,
                'device': self.device
            }
            
            torch.save(checkpoint, filepath)
            print(f"模型已保存到: {filepath}")

    # --------- 训练 & 验证 ---------
    def fit_with_visual(self, X_train, y_train, X_valid, y_valid, epochs=50, patience=7):
        """
        带可视化 + 早停 的训练接口
        epochs  默认 50 轮
        patience 默认 7 轮无提升就停
        """
        os.makedirs('output/model', exist_ok=True)
        os.makedirs('output/img',   exist_ok=True)

        n_features = X_train.shape[1]
        self.model = CNN1D(n_features).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_loader = self.df_to_loader(X_train, y_train, shuffle=True)
        valid_loader = self.df_to_loader(X_valid, y_valid, shuffle=False)
        self.n_features = X_train.shape[1]  # 记录输入特征维度

        pos_weight = self.calc_pos_weight(y_train)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))


        history = {k: [] for k in
                ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                    'train_pre', 'val_pre', 'train_rec', 'val_rec']}
        early_stopping = EarlyStopping(patience=patience, save_path='output/model/best_cnn.pt')

        for epoch in range(1, epochs + 1):
            # ===== 训练 =====
            self.model.train()
            train_loss, train_true, train_score = 0., [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                score = self.model(xb)
                loss  = self.criterion(score, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_true.append(yb.cpu().numpy())
                train_score.append(score.detach().cpu().numpy())
            train_true = np.concatenate(train_true)
            train_pred = (np.concatenate(train_score) > 0.5).astype(int)

            # ===== 验证 =====
            self.model.eval()
            val_loss, val_true, val_score = 0., [], []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    score = self.model(xb)
                    loss  = self.criterion(score, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_true.append(yb.cpu().numpy())
                    val_score.append(score.cpu().numpy())
            val_true = np.concatenate(val_true)
            val_pred = (np.concatenate(val_score) > 0.5).astype(int)

            # ===== 指标 =====
            def calc(true, pred):
                return (accuracy_score(true, pred),
                        precision_score(true, pred, zero_division=0),
                        recall_score(true, pred, zero_division=0))

            train_acc, train_pre, train_rec = calc(train_true, train_pred)
            val_acc,   val_pre,   val_rec   = calc(val_true,   val_pred)
            train_loss /= len(train_loader.dataset)
            val_loss   /= len(valid_loader.dataset)

            # 记录
            for k in history:
                history[k].append(locals()[k])
            self._plot_metrics(history, epoch)

            # ===== 早停判断 =====
            val_auc = roc_auc_score(val_true, np.concatenate(val_score))
            print(f'Epoch {epoch:02d} | '
                f'train loss {train_loss:.4f} acc {train_acc:.4f} pre {train_pre:.4f} rec {train_rec:.4f} | '
                f'val loss {val_loss:.4f} acc {val_acc:.4f} pre {val_pre:.4f} rec {val_rec:.4f} | '
                f'val AUC {val_auc:.4f}')
            if early_stopping(val_auc, self.model):
                print(f'Early stopped at epoch {epoch}')
                break

        # 训练结束后加载最优权重
        self.model.load_state_dict(torch.load('output/model/best_cnn.pt'))
        self.save_model('output/model/cnn_trained.pth')
        print('Finished, best AUC:', early_stopping.best_auc)

    def _plot_metrics(self, history, epoch):
        """
        绘制 4 个子图：损失、准确率、精确率、召回率
        并保存到 output/img/metrics.png
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 中文/英文标签均可，这里用中文
        sub_titles = [
            '训练和验证损失',
            '训练和验证准确率',
            '训练和验证精确率',
            '训练和验证召回率'
        ]
        # 曲线在图例中显示的名称
        label_map = {'train': '训练', 'val': '验证'}

        # 4 组指标对应 4 个子图
        plot_keys = [
            ['train_loss', 'val_loss'],
            ['train_acc',  'val_acc'],
            ['train_pre',  'val_pre'],
            ['train_rec',  'val_rec']
        ]
        y_labels  = ['Loss', 'Accuracy', 'Precision', 'Recall']

        for ax, title, keys, y_lab in zip(axes, sub_titles, plot_keys, y_labels):
            for k in keys:
                ax.plot(history[k], label=label_map[k.split('_')[0]], marker='o', markersize=3)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(y_lab)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('output/img/cnn_training_history.png', dpi=150)
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
        plt.savefig('output/img/cnn_threshold_optimization.png', dpi=150, bbox_inches='tight')
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

    # --------- 预测 ---------
    def predict_one(self, x_single, y_true=None, best_threshold=0.5):
        """
        单条样本预测 & 与真实值对比
        参数
        ----
        x_single : pd.Series / 1-D np.ndarray / list
            一条样本的特征，长度必须等于 n_features
        y_true : int/float, optional
            真实标签，只做对比展示，不参与运算

        返回
        ----
        prob : float
            模型输出的正类概率
        pred_label : int
            0 或 1
        """
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

    def _plot_confusion_matrix_minimal(self, X_test, y_test, best_threshold):
        """最小化混淆矩阵绘制"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        
        # 收集预测结果
        y_pred = []
        for i in range(min(2500, len(X_test))):  # 限制数量避免太长
            _, pred = self.predict_one(x_single = X_test.iloc[i], best_threshold=best_threshold)
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
        plt.savefig('output/img/cnn_confusion_matrix.png', dpi=150)
        plt.close()

    # --------- 加载模型 ---------
    def load_model(self, filepath='output/model/cnn_model.pth'):
        """
        加载已保存的模型
        
        参数:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载检查点
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复训练信息
        self.n_features = checkpoint['n_features']
        self.lr = checkpoint['lr']
        self.pos_weight = checkpoint['pos_weight']
        
        # 重新初始化模型
        self.model = CNN1D(self.n_features).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"模型已从 {filepath} 加载")
        print(f"模型信息: 输入维度={self.n_features}, 学习率={self.lr}")

if __name__ == '__main__':  
    X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')
    cnn = CNNModel()
    cnn.fit_with_visual(X_train, y_train, X_valid, y_valid, epochs=50)

    # 测试集表现
    cnn.model.load_state_dict(torch.load('output/model/best_cnn.pt'))
    # cnn.load_model('output/model/cnn_trained.pth')
    best_threshold = cnn.optimize_threshold(X_valid, y_valid)
    cnn._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)
    
