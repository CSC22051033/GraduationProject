# ====== 数据预处理 ======
from carClaims import split     # 数据集划分 - 分割原始数据集为训练集、验证集和测试集
from carClaims import Model, EarlyStopping

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EarlyStoppingF1:
    """早停：监控验证 F1，patience 内无提升就停"""
    def __init__(self, patience=3, verbose=True, save_path='best.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_f1 = 0.0
        self.save_path = save_path

    def __call__(self, val_f1, model):
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f'EarlyStopping: New best F1 = {val_f1:.4f}, model saved.')
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: patience = {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                return True
            return False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)  # 调整维度
        return self.dropout(x)

class ImprovedTransformer(nn.Module):
    def __init__(self, n_features: int, vocab_size: int, seq_len: int = 50):
        super(ImprovedTransformer, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # 嵌入层 - 针对分类特征
        self.embedding = nn.Embedding(vocab_size + 1, 64, padding_idx=0)  # +1 for padding
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=64, dropout=0.3)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 输出层
        self.fc2 = nn.Linear(32, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        x形状: (batch_size, sequence_length) 或 (batch_size, n_features)
        """
        if x.dim() == 2 and x.dtype != torch.long:
            x = x.long()
        
        # 嵌入层
        x = self.embedding(x) * np.sqrt(64)  # 缩放嵌入
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(x)
        
        # 使用平均池化
        pooled = transformer_out.mean(dim=1)
        
        # 全连接层
        pooled = self.bn1(pooled)
        pooled = self.relu(self.fc1(pooled))
        pooled = self.bn2(pooled)
        pooled = self.dropout(pooled)
        
        output = self.fc2(pooled)
        return output

class SimplifiedTransformer(nn.Module):
    """更简单的模型，专门针对表格数据"""
    def __init__(self, n_features, vocab_size, seq_len=50):
        super().__init__()
        
        # 更小的嵌入维度
        self.embedding = nn.Embedding(vocab_size + 1, 16, padding_idx=0)
        
        # 移除位置编码（表格数据不需要）
        
        # 更简单的特征提取
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 更小的全连接层
        self.fc1 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 嵌入
        x = self.embedding(x.long()).transpose(1, 2)  # (batch, channels, seq_len)
        
        # 卷积特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 池化
        x = self.global_pool(x).squeeze(-1)
        
        # 分类
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TransformerModel(Model):
    def __init__(self, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.encoders = {}

    def _label_to_01(self, y):
        """
        将任意形式的标签转成 0/1 浮点数组。
        支持：Series、ndarray、list、字符串、数字。
        """
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y)

        # 如果是字符串类别，先做 LabelEncoder
        if y.dtype.kind in {'U', 'O'}:          # unicode / object
            if not hasattr(self, '_label_encoder'):
                self._label_encoder = LabelEncoder()
                self._label_encoder.fit(y)
            y = self._label_encoder.transform(y)

        # 确保是 0/1 形式
        y = y.astype(np.float32)
        return y

    def calc_pos_weight(self, y_train):
        y_np = self._label_to_01(y_train)       # 先转 0/1
        n_pos = (y_np == 1).sum()
        n_neg = (y_np == 0).sum()
        if n_pos == 0:
            return torch.tensor([1.0]).float().to(self.device)
        return torch.tensor([n_neg / n_pos]).float().to(self.device)
    
    def df_to_loader(self, X, y=None, shuffle=False, batch_size=64):
        """将DataFrame转换为DataLoader"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 转换数据为tensor，确保数值类型
        X_tensor = torch.FloatTensor(X.astype(np.float32))
        
        if y is not None:
            # 确保y是数值类型
            if hasattr(y, 'values'):
                y_np = y.values
            else:
                y_np = np.array(y)
            y_tensor = torch.FloatTensor(y_np.astype(np.float32))
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
       
    def _plot_metrics(self, hist):
        """修复的绘图函数 - 确保所有需要的键都存在"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 使用中文小标题
        metrics = [('训练和验证损失', 'train_loss', 'val_loss'),
                   ('训练和验证准确率', 'train_acc', 'val_acc'),
                   ('训练和验证精确率', 'train_pre', 'val_pre'),
                   ('训练和验证召回率', 'train_rec', 'val_rec')]
        
        # 中文图例
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
                
                # y轴标签根据指标类型设置
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
        os.makedirs('output/img', exist_ok=True)
        plt.savefig('output/img/transformer_training_history.png', dpi=150)
        plt.close(fig)
    
    def df_to_tensor(self, df, seq_len=None):
        """将DataFrame转换为序列张量"""
        if seq_len is None:
            seq_len = self.seq_len
        
        # 创建编码器（如果需要）
        if not self.encoders:
            for col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.encoders[col] = le
        
        # 逐行编码
        ids = []
        for _, row in df.iterrows():
            seq = []
            for col in df.columns:
                le = self.encoders[col]
                val = str(row[col])
                if val in le.classes_:
                    seq.append(le.transform([val])[0] + 1)  # +1 把 0 留给 PAD
                else:
                    seq.append(0)  # 未知类别用0填充
            
            # 截断 / 填充
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            else:
                seq = seq + [0] * (seq_len - len(seq))
            ids.append(seq)
        
        return torch.LongTensor(ids)
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, verbose=True):
        torch.manual_seed(42)
        self.n_features = X_train.shape[1]

        # ****** 新增：把 y 转成 0/1 ******
        y_train_01 = self._label_to_01(y_train)
        y_valid_01 = self._label_to_01(y_valid)
        self.pos_weight = self.calc_pos_weight(y_train_01)
        # **********************************

        # 后续全部用 y_train_01 / y_valid_01 代替原来的 y_train / y_valid
        print("正在将数据转换为序列...")
        X_train_seq = self.df_to_tensor(X_train)
        X_valid_seq = self.df_to_tensor(X_valid)
        vocab_size = int(torch.max(torch.cat([X_train_seq, X_valid_seq]))) + 1

        class SequenceDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences = sequences
                self.labels = labels
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

        train_dataset = SequenceDataset(X_train_seq, torch.FloatTensor(y_train_01))
        valid_dataset = SequenceDataset(X_valid_seq, torch.FloatTensor(y_valid_01))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        
        # 4. 模型、优化器、早停
        self.model = SimplifiedTransformer(
            n_features=self.n_features,
            vocab_size=vocab_size,
            seq_len=self.seq_len
        ).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3) 
        
        os.makedirs('output/model', exist_ok=True)
        early_stop = EarlyStoppingF1(patience=3, verbose=verbose,
                             save_path='output/model/best_transformer.pt')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2)
        
        # 初始化历史记录
        history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'train_pre': [], 'val_pre': [],
            'train_rec': [], 'val_rec': []
        }
        
        # 5. 训练循环
        for epoch in range(1, epochs + 1):
            # ---------- 训练 ----------
            self.model.train()
            train_loss, train_preds, train_labels = 0.0, [], []
            
            for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} - train', disable=not verbose):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * xb.size(0)
                train_preds.append(torch.sigmoid(out).detach().cpu().numpy())
                train_labels.append(yb.cpu().numpy())
            
            train_loss /= len(train_loader.dataset)
            train_preds_np = np.concatenate(train_preds).flatten()
            train_labels_np = np.concatenate(train_labels)
            
            # 计算训练集指标
            if len(np.unique(train_labels_np)) > 1:
                train_auc = roc_auc_score(train_labels_np, train_preds_np)
                train_preds_binary = (train_preds_np > 0.5).astype(int)
                train_acc = accuracy_score(train_labels_np, train_preds_binary)
                train_pre = precision_score(train_labels_np, train_preds_binary, zero_division=0)
                train_rec = recall_score(train_labels_np, train_preds_binary, zero_division=0)
            else:
                train_auc = train_acc = train_pre = train_rec = 0.0
            
            # ---------- 验证 ----------
            val_loss, val_probs, val_labels = self._evaluate(valid_loader)
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = accuracy_score(val_labels, val_preds)
            val_pre = precision_score(val_labels, val_preds, zero_division=0)
            val_rec = recall_score(val_labels, val_preds, zero_division=0)
            val_f1  = f1_score(val_labels, val_preds, zero_division=0)
            scheduler.step(val_f1)
            
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
            if early_stop(val_f1, self.model):
                if verbose:
                    print(f'Early stop at epoch {epoch}')
                break
        
        # 6. 加载最优权重
        self.model.load_state_dict(torch.load('output/model/best_transformer.pt', map_location=self.device))
        if verbose:
            print('Loaded best model.')
        
        # 7. 保存完整模型
        self.save('output/model/transformer_model.pth')
        if verbose:
            print('Saved full model to output/model/transformer_model.pth')
        
        # 8. 绘制训练历史
        self._plot_metrics(history)
        return history
    
    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb.unsqueeze(1))
                total_loss += loss.item() * xb.size(0)
                preds.append(torch.sigmoid(out).cpu().numpy())
                labels.append(yb.cpu().numpy())
        total_loss /= len(data_loader.dataset)
        preds_np = np.concatenate(preds).flatten()
        labels_np = np.concatenate(labels)
        return total_loss, preds_np, labels_np   # 返回概率和标签

    def predict(self, x_single, y_true=None, best_threshold=0.5):
        # 1. 转换输入数据
        if isinstance(x_single, pd.Series):
            x_single = pd.DataFrame([x_single])
        elif isinstance(x_single, np.ndarray):
            x_single = pd.DataFrame(x_single.reshape(1, -1), columns=[f'col_{i}' for i in range(x_single.shape[0])])
        
        # 2. 转换为序列
        x_seq = self.df_to_tensor(x_single)
        
        # 3. 预测
        self.model.eval()
        with torch.no_grad():
            x_seq = x_seq.to(self.device)
            logit = self.model(x_seq)
            prob = torch.sigmoid(logit).cpu().item()
        
        pred_label = int(prob > best_threshold)
        
        # 4. 打印对比
        if y_true is not None:
            # 把标量/列表统一转成 1-D 列表
            y_true_01 = self._label_to_01([y_true])      # ← 包一层列表
            print(f'真实标签: {int(y_true_01[0])}  |  预测概率: {prob:.4f}  |  预测类别: {pred_label}')
        return prob, pred_label
    
    def save(self, filepath='output/model/transformer_model.pth'):
        """保存完整模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'n_features': self.n_features,
            'seq_len': self.seq_len,
            'encoders': self.encoders,
            'pos_weight': self.pos_weight.cpu() if hasattr(self, 'pos_weight') else None
        }
        torch.save(save_dict, filepath)
        print(f'Model saved to {filepath}')
    
    def load(self, filepath='output/model/transformer_model.pth'):
        """加载完整模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 重建模型
        self.n_features = checkpoint['n_features']
        self.seq_len = checkpoint['seq_len']
        self.encoders = checkpoint['encoders']
        
        # 计算词汇表大小
        vocab_size = 0
        for le in self.encoders.values():
            vocab_size = max(vocab_size, len(le.classes_))
        vocab_size += 1  # 为padding留位置
        
        self.model = ImprovedTransformer(
            n_features=self.n_features,
            vocab_size=vocab_size,
            seq_len=self.seq_len
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载权重参数
        if checkpoint['pos_weight'] is not None:
            self.pos_weight = checkpoint['pos_weight'].to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        print(f'Model loaded from {filepath}')
        return self
    
    def _plot_confusion_matrix_minimal(self, X_test, y_test, best_threshold):
        """最小化混淆矩阵绘制"""
        # 转换测试数据
        X_test_seq = self.df_to_tensor(X_test)
        y_test_01 = self._label_to_01(y_test)
        y_test_t = torch.FloatTensor(y_test_01)
        
        # 创建DataLoader
        test_dataset = torch.utils.data.TensorDataset(X_test_seq, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 收集预测结果
        y_true_list, y_pred_list = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                preds = (probs > best_threshold).astype(int)
                y_pred_list.extend(preds)
                y_true_list.extend(yb.numpy())
        
        # 计算并绘制混淆矩阵
        cm = confusion_matrix(y_true_list, y_pred_list)
        
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
        os.makedirs('output/img', exist_ok=True)
        plt.savefig('output/img/transformer_confusion_matrix.png', dpi=150)
        plt.close()
    
    def optimize_threshold(self, X_valid, y_valid, step=0.01):
        """阈值优化函数"""
        # 转换验证数据
        X_valid_seq = self.df_to_tensor(X_valid)
        y_valid_01 = self._label_to_01(y_valid)
        y_valid_t = torch.FloatTensor(y_valid_01)
        
        # 创建DataLoader
        valid_dataset = torch.utils.data.TensorDataset(X_valid_seq, y_valid_t)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        
        # 获取预测概率
        self.model.eval()
        y_true_list, y_score_list = [], []
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                y_score_list.extend(probs)
                y_true_list.extend(yb.numpy())
        
        y_true = np.array(y_true_list)
        y_scores = np.array(y_score_list)
        
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
        plt.savefig('output/img/transformer_threshold_optimization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 展示优化结果
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

# 使用示例
def main():
    # 假设split函数已定义
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    data = pd.read_csv('carclaims.csv')
    X = data.drop('FraudFound', axis=1)
    y = data['FraudFound']
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 创建并训练模型
    model = TransformerModel(seq_len=50)
    history = model.fit(X_train, y_train, X_valid, y_valid, epochs=30, verbose=True)
    
    # 优化阈值
    best_threshold = model.optimize_threshold(X_valid, y_valid)
    
    # 绘制混淆矩阵
    model._plot_confusion_matrix_minimal(X_test, y_test, best_threshold)
    
    # 单个样本预测
    sample_idx = 0
    prob, pred = model.predict(X_test.iloc[sample_idx], y_test.iloc[sample_idx], best_threshold)
    
    return model, history

if __name__ == "__main__":
    model, history = main()