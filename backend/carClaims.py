# ====== 基础环境 ======
import os               # 操作系统接口：文件/目录操作、环境变量读取等
import numpy as np      # 数值计算核心库，提供高效多维数组及矩阵运算
import pandas as pd     # 表格型数据处理利器，DataFrame 是其核心数据结构
# ====== 通用工具 ======
from collections import Counter     # 快速统计可迭代对象中各元素出现次数（常用于查看类别分布）
# ====== 样本不平衡处理 ======
from imblearn.over_sampling import SMOTE                    # 合成少数类过采样：凭空生成少数类样本，减缓类别失衡
from imblearn.under_sampling import RandomUnderSampler      # 随机多数类欠采样：随机删减多数类样本
# ====== 特征缩放 ======
from sklearn.preprocessing import MinMaxScaler      # 最小-最大归一化：把数值特征压缩到 [0, 1] 区间
from sklearn.preprocessing import StandardScaler
# ====== 数据集划分 ======
from sklearn.model_selection import train_test_split
# ====== 特征选择 ======
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
# ====== PyTorch深度学习框架 ======
import torch                      # PyTorch深度学习框架，提供张量计算和自动求导功能
import torch.nn as nn             # PyTorch神经网络模块，提供网络层、损失函数等
from torch.utils.data import TensorDataset, DataLoader    # 数据集封装和数据加载工具
# ====== 可视化 ======
import matplotlib.pyplot as plt     # 绘图库，用于数据分布、结果曲线等可视化
# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.utils import class_weight

class CarClaimsPreprocessor:
    """
        该类用于实现数据预处理
    """
    def __init__(self, file_path):
        """
        初始化预处理器
        Args:
            file_path: CSV文件路径
        """
        self.file_path = file_path
        self.useless_columns = ['PolicyNumber', 'PolicyType']
        # "PolicyNumber"特征仅仅是保单的编号，对于后续的车险欺诈识别建模并没有实际的预测意义。因此，将该特征从数据集中删除，以简化模型的输入，提高训练效率。
        # "Policy Type"特征是"BasePolicy"和"VehicleCategory"特征的组合，记录了车辆类别和保单类型。 所以删除"Policy Type"重复特征。

        # 创建输出文件夹
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"数据加载成功！数据形状: {self.df.shape}")
            print("\n数据前5行:")
            print(self.df.head())
            print("\n数据基本信息:")
            print(self.df.info())
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    # 以下为数据预处理的异常处理部分
    # 1._preliminary_outlier_detection：数值型异常值检测（IQR方法）
    # 2.handle_categorical_anomalies：处理分类变量的异常值
    # 3.handle_age_anomalies：处理数值型变量的异常值
    def _preliminary_outlier_detection(self):
        """初步异常值检测"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("\n没有数值型列可用于异常值检测")
            return
            
        print("\n初步异常值检测 (IQR方法):")
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_percent = (len(outliers) / len(self.df)) * 100
            
            print(f"{col}: {len(outliers)} 个异常值 ({outlier_percent:.2f}%)")

    def handle_categorical_anomalies(self):
        """
        处理分类变量的异常值：
        DayOfWeekClaimed列的异常值用出现最多次数的'Monday'代替,
        MonthClaimed列的异常值用出现最多次数的'Jan'代替。
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        print("=" * 50)
        print("处理分类变量异常值")
        print("=" * 50)
        
        # 处理DayOfWeekClaimed列的异常值
        if 'DayOfWeekClaimed' in self.df.columns:
            # 检查是否存在字符串'0'作为异常值
            if '0' in self.df['DayOfWeekClaimed'].values:
                # 找到出现最频繁的值（排除'0'）
                most_frequent_day = self.df[self.df['DayOfWeekClaimed'] != '0']['DayOfWeekClaimed'].mode()
                if len(most_frequent_day) > 0:
                    replacement_day = most_frequent_day[0]
                    anomalies_count = (self.df['DayOfWeekClaimed'] == '0').sum()
                    self.df.loc[self.df['DayOfWeekClaimed'] == '0', 'DayOfWeekClaimed'] = replacement_day
                    print(f"DayOfWeekClaimed列: 替换了 {anomalies_count} 个异常值('0')为 '{replacement_day}'")
                else:
                    print("DayOfWeekClaimed列: 没有找到有效的替换值")
            else:
                print("DayOfWeekClaimed列: 未发现异常值('0')")
        
        # 处理MonthClaimed列的异常值
        if 'MonthClaimed' in self.df.columns:
            # 检查是否存在字符串'0'作为异常值
            if '0' in self.df['MonthClaimed'].values:
                # 找到出现最频繁的值（排除'0'）
                most_frequent_month = self.df[self.df['MonthClaimed'] != '0']['MonthClaimed'].mode()
                if len(most_frequent_month) > 0:
                    replacement_month = most_frequent_month[0]
                    anomalies_count = (self.df['MonthClaimed'] == '0').sum()
                    self.df.loc[self.df['MonthClaimed'] == '0', 'MonthClaimed'] = replacement_month
                    print(f"MonthClaimed列: 替换了 {anomalies_count} 个异常值('0')为 '{replacement_month}'")
                else:
                    print("MonthClaimed列: 没有找到有效的替换值")
            else:
                print("MonthClaimed列: 未发现异常值('0')")
    
    def handle_age_anomalies(self):
        """
        处理Age列的异常值0，用除0外该列平均值代替
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        print("=" * 50)
        print("处理Age列异常值")
        print("=" * 50)
        
        if 'Age' in self.df.columns:
            # 计算除0外的平均值
            age_mean = self.df[self.df['Age'] != 0]['Age'].mean()
            anomalies_count = (self.df['Age'] == 0).sum()
            
            # 替换异常值
            self.df.loc[self.df['Age'] == 0, 'Age'] = int(age_mean)
            
            print(f"Age列: 替换了 {anomalies_count} 个异常值(0)为平均值 {age_mean:.2f}")
        else:
            print("Age列不存在于数据中")
    
    def convert_yes_no_to_bool(self):
        """
        将包含'Yes','No'的列转换为布尔值True,False
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        print("=" * 50)
        print("转换Yes/No为布尔值")
        print("=" * 50)
        
        converted_columns = []
        
        for col in self.df.columns:
            # 检查列是否包含'Yes'和'No'值
            if self.df[col].dtype == 'object':
                unique_values = self.df[col].unique()
                if set(unique_values) == {'Yes', 'No'} or set(unique_values) == {'yes', 'no'} or set(unique_values) == {'YES', 'NO'}:
                    # 转换为布尔值
                    self.df[col] = self.df[col].map({'Yes': True, 'No': False, 'yes': True, 'no': False, 'YES': True, 'NO': False})
                    converted_columns.append(col)
                    print(f"转换列 '{col}': Yes/No -> True/False")
        
        if not converted_columns:
            print("没有找到包含Yes/No值的列")
        else:
            print(f"总共转换了 {len(converted_columns)} 个列: {converted_columns}")
    
    def remove_useless_features(self, useless_columns=None):
        """
        去除无用特征
        
        Args:
            useless_columns: 手动指定的无用特征列名列表
        
        Returns:
            removed_columns: 被移除的列名列表
        """
        if not hasattr(self, 'df'):
            print("请先加载数据！")
            return []
        
        original_shape = self.df.shape
        removed_columns = []

        if useless_columns is None:
            useless_columns = self.useless_columns
        
        # 手动指定的无用特征
        if useless_columns:
            useless_columns = [col for col in useless_columns if col in self.df.columns]
            self.df = self.df.drop(columns=useless_columns)
            removed_columns.extend(useless_columns)
            print("\n")
            print("=" * 50)
            print(f"移除手动指定的无用特征: {useless_columns}")   
        
        # 输出结果
        new_shape = self.df.shape
        print(f"\n特征移除完成:")
        print(f"原始数据形状: {original_shape}")
        print(f"新数据形状: {new_shape}")
        print(f"移除的特征数量: {len(removed_columns)}")
        print(f"被移除的特征: {removed_columns}")
        
        return removed_columns
    
    def one_hot_encoding(self):
        """
        对非数值列进行独热编码
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        print("=" * 50)
        print("进行独热编码")
        print("=" * 50)
        
        # 获取非数值列
        non_numeric_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(non_numeric_cols) == 0:
            print("没有非数值列需要编码")
            return self.df
        
        print(f"将对以下 {len(non_numeric_cols)} 个非数值列进行独热编码:")
        print(list(non_numeric_cols))
        
        # 进行独热编码
        encoded_df = pd.get_dummies(self.df, columns=non_numeric_cols, prefix=non_numeric_cols, drop_first=False)
        
        # 更新原始数据框
        original_shape = self.df.shape
        self.df = encoded_df
        new_shape = self.df.shape
        
        print(f"独热编码完成！")
        print(f"原始数据形状: {original_shape}")
        print(f"编码后数据形状: {new_shape}")
        print(f"新增列数: {new_shape[1] - original_shape[1]}")
        
        return self.df
    
    def balance_dataset(self, target_column, random_state=42, smoteRate=0.6, underRate=0.65):
        """
        平衡数据集，处理类别不平衡问题
        
        Args:
            target_column: 目标变量列名
            random_state: 随机种子
            
        Returns:
            balanced_df: 平衡后的数据框
        """
        if self.df is None:
            print("请先加载数据！")
            return None
        
        if target_column not in self.df.columns:
            print(f"目标列 '{target_column}' 不存在于数据中")
            return None
        
        print("=" * 50)
        print(f"使用 smote_under 方法平衡数据集")
        print("=" * 50)
        
        # 分离特征和目标变量
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # 显示原始类别分布
        original_dist = Counter(y)
        print(f"原始数据类别分布: {dict(original_dist)}")
        
        # 计算不平衡比例
        majority_class = max(original_dist, key=original_dist.get)
        minority_class = min(original_dist, key=original_dist.get)
        imbalance_ratio = original_dist[majority_class] / original_dist[minority_class]
        print(f"不平衡比例: {imbalance_ratio:.2f}:1")

        # SMOTE过采样 + 随机欠采样组合
        # 先使用SMOTE过采样少数类
        smote = SMOTE(random_state=random_state, sampling_strategy=smoteRate)  # sampling_strategy
        X_resampled, y_resampled = smote.fit_resample(X, y)
            
        # 再使用随机欠采样多数类
        under_sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=underRate)  # sampling_strategy
        X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)
        print("使用 SMOTE + RandomUnderSampler 组合方法进行平衡")

         # 显示平衡后的类别分布
        balanced_dist = Counter(y_resampled)
        print(f"平衡后数据类别分布: {dict(balanced_dist)}")
        
        # 创建平衡后的数据框
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df = balanced_df.copy()  # 创建副本以避免碎片化
        balanced_df[target_column] = y_resampled
        
        # 可视化类别分布
        self._plot_class_distribution(original_dist, balanced_dist)
        
        print(f"\n平衡完成！")
        print(f"原始数据形状: {self.df.shape}")
        print(f"平衡后数据形状: {balanced_df.shape}")
        
        self.df = balanced_df
        return balanced_df
    
    def _plot_class_distribution(self, original_dist, balanced_dist, method='smote_under'):
        """
        绘制类别分布对比图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始分布
        classes = list(original_dist.keys())
        counts_original = list(original_dist.values())
        ax1.bar(classes, counts_original, color=['lightblue', 'lightcoral'])
        ax1.set_title('原始数据类别分布')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('样本数量')
        for i, v in enumerate(counts_original):
            ax1.text(i, v, str(v), ha='center', va='bottom')
        
        # 平衡后分布
        counts_balanced = [balanced_dist.get(cls, 0) for cls in classes]
        ax2.bar(classes, counts_balanced, color=['lightblue', 'lightcoral'])
        ax2.set_title(f'平衡后数据类别分布 ({method})')
        ax2.set_xlabel('类别')
        ax2.set_ylabel('样本数量')
        for i, v in enumerate(counts_balanced):
            ax2.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir+'/img', f'class_distribution_{method}.png'), dpi=300, bbox_inches='tight')
        # plt.show()

    def convert_boolean_to_numeric(self, exclude_columns=None):
        """
        将布尔类型列转换为数值类型
        
        Args:
            exclude_columns: 要排除的列名列表，不进行转换
            
        Returns:
            转换后的DataFrame
        """
        if self.df is None:
            print("请先加载数据！")
            return None
        
        print("=" * 50)
        print("布尔类型列转换为数值类型")
        print("=" * 50)
        
        # 确定所有布尔列
        all_boolean_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        if len(all_boolean_cols) == 0:
            print("没有找到布尔类型列")
            return self.df
        
        # 确定要处理的布尔列
        boolean_cols_to_process = all_boolean_cols.copy()
        
        # 排除指定的列
        if exclude_columns:
            original_count = len(boolean_cols_to_process)
            boolean_cols_to_process = [
                col for col in boolean_cols_to_process 
                if col not in exclude_columns
            ]
            excluded_count = original_count - len(boolean_cols_to_process)
            if excluded_count > 0:
                print(f"排除 {excluded_count} 个布尔列: {[col for col in exclude_columns if col in all_boolean_cols]}")
        
        if len(boolean_cols_to_process) == 0:
            print("没有需要转换的布尔列")
            # 显示所有布尔列的状态
            if all_boolean_cols:
                print(f"\n数据集包含以下 {len(all_boolean_cols)} 个布尔列:")
                for col in all_boolean_cols:
                    status = "排除" if exclude_columns and col in exclude_columns else "保留"
                    print(f"  - {col}: {status}")
            return self.df
        
        # 显示转换信息
        print(f"将转换以下 {len(boolean_cols_to_process)} 个布尔列:")
        for i, col in enumerate(boolean_cols_to_process, 1):
            unique_values = self.df[col].dropna().unique()
            true_count = (self.df[col] == True).sum()
            false_count = (self.df[col] == False).sum()
            null_count = self.df[col].isnull().sum()
            print(f"  {i}. {col}: True={true_count}, False={false_count}, 缺失值={null_count}")
        
        # 显示被排除的布尔列（如果有）
        if exclude_columns:
            excluded_boolean_cols = [col for col in exclude_columns if col in all_boolean_cols]
            if excluded_boolean_cols:
                print(f"\n已排除以下 {len(excluded_boolean_cols)} 个布尔列:")
                for col in excluded_boolean_cols:
                    print(f"  - {col}")
        
        converted_df = self.df.copy()
        
        # 转换布尔列为数值类型
        success_count = 0
        failed_cols = []
        
        for col in boolean_cols_to_process:
            try:
                # 保存原始值的统计信息（转换前）
                true_count = (converted_df[col] == True).sum()
                false_count = (converted_df[col] == False).sum()
                null_count = converted_df[col].isnull().sum()
                
                # 转换为数值类型
                converted_df[col] = converted_df[col].astype(float)
                
                # 验证转换结果
                after_true_count = (converted_df[col] == 1.0).sum()
                after_false_count = (converted_df[col] == 0.0).sum()
                after_null_count = converted_df[col].isnull().sum()
                
                # 检查转换是否正确
                if (true_count == after_true_count and 
                    false_count == after_false_count and
                    null_count == after_null_count):
                    print(f"✓ 列 '{col}': 成功转换 (True={true_count}, False={false_count}, 缺失值={null_count})")
                    success_count += 1
                else:
                    print(f"⚠ 列 '{col}': 转换结果不一致，需要检查")
                    failed_cols.append(col)
                    
            except Exception as e:
                print(f"✗ 列 '{col}' 转换失败: {e}")
                failed_cols.append(col)
        
        # 更新原始数据框
        self.df = converted_df
        
        # 显示转换结果总结
        print(f"\n{'='*50}")
        print("布尔类型转换完成！")
        print(f"{'='*50}")
        print(f"成功转换列数: {success_count}/{len(boolean_cols_to_process)}")
        
        if failed_cols:
            print(f"转换失败列数: {len(failed_cols)}")
            print("失败列列表:", failed_cols)
        
        return converted_df

    def minmax(self):
        """最小最大归一化"""
        num_cols = self.df.select_dtypes(include='number', exclude='bool').columns.tolist()
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

    def processor(self):
        # 加载数据
        self.load_data()
        # 数据替换（Yes/No -> True/False）
        self.convert_yes_no_to_bool()
        # 数据异常值处理
        self._preliminary_outlier_detection()
        self.handle_categorical_anomalies()
        self.handle_age_anomalies()
        # 去除无用特征
        self.remove_useless_features()
        # 数据编码
        self.one_hot_encoding()     # 独热编码
        # 数据平衡
        self.balance_dataset(target_column='FraudFound')
        # 数据替换（True/False -> 1/0）
        self.convert_boolean_to_numeric(exclude_columns='FraudFound')
        # 数据归一化
        self.minmax()
        return self.df

class FeatureSelect:
    """特征选择"""
    def __init__(self, save_path='output/'):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'img'), exist_ok=True)

        # 保存训练过程中产生的对象
        self.scaler = None          # 用于 CNN 的标准化器
        self.cnn_feature_extractor = None  # CNN 特征提取子模型
        self.rf = None               # 随机森林模型
        self.feature_importances_ = None  # 特征重要性
        self.top_features_ = None     # 选出的特征名（可选）

    def fit(self, X_train, y_train,
            use_cnn=True, cnn_output_dim=64, cnn_epochs=20, cnn_batch_size=32,
            rf_n_estimators=300, rf_max_depth=None):
        """
        在训练集上拟合特征选择器
        """
        self.use_cnn = use_cnn
        self.cnn_output_dim = cnn_output_dim if use_cnn else 0

        # 如果需要 CNN，则训练并提取深度特征（仅对训练集）
        if use_cnn:
            # 1. 标准化
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # 2. 训练 CNN 并提取特征提取器
            self.cnn_feature_extractor = self._train_cnn(
                X_train_scaled, y_train,
                output_dim=cnn_output_dim,
                epochs=cnn_epochs,
                batch_size=cnn_batch_size
            )

            # 3. 对训练集提取深度特征
            X_train_cnn = self._extract_cnn_features(X_train_scaled)
            # 将深度特征转换为 DataFrame（列名统一）
            df_cnn = pd.DataFrame(X_train_cnn,
                                   columns=[f'deep_feat_{i+1}' for i in range(cnn_output_dim)])
            # 拼接原始特征和深度特征
            X_train_processed = pd.concat([X_train.reset_index(drop=True), df_cnn], axis=1)
        else:
            X_train_processed = X_train.copy()

        # 训练随机森林
        self.rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf.fit(X_train_processed, y_train)

        # 保存特征重要性
        self.feature_importances_ = pd.Series(
            self.rf.feature_importances_,
            index=X_train_processed.columns
        ).sort_values(ascending=False)

        return self

    def transform(self, X):
        """
        对任意数据集 X 应用同样的变换（标准化 + 深度特征提取 + 保持列一致）
        """
        if self.use_cnn:
            # 1. 标准化（使用训练集的 scaler）
            X_scaled = self.scaler.transform(X)

            # 2. 提取深度特征
            X_cnn = self._extract_cnn_features(X_scaled)
            df_cnn = pd.DataFrame(X_cnn,
                                   columns=[f'deep_feat_{i+1}' for i in range(self.cnn_output_dim)])

            # 3. 拼接
            X_processed = pd.concat([X.reset_index(drop=True), df_cnn], axis=1)
        else:
            X_processed = X.copy()

        # 确保列顺序与训练时一致（可选）
        # X_processed = X_processed[self.feature_importances_.index]  # 若需要按重要性排序
        return X_processed

    def fit_transform(self, X_train, y_train, **kwargs):
        """方便直接调用 fit + transform on training set"""
        self.fit(X_train, y_train, **kwargs)
        return self.transform(X_train)

    def _train_cnn(self, X_train_scaled, y_train, output_dim, epochs, batch_size):
        """训练 CNN 并返回特征提取器（子模型）"""
        n_features = X_train_scaled.shape[1]
        X_train_reshaped = X_train_scaled.reshape(-1, n_features, 1)

        # 构建模型（与原来类似）
        inputs = Input(shape=(n_features, 1))
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        dense_layer = Dense(output_dim, activation='relu', name='deep_features')(x)
        x = Dropout(0.5)(dense_layer)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # 处理类别权重
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        model.fit(X_train_reshaped, y_train,
                  epochs=epochs, batch_size=batch_size,
                  class_weight=class_weight_dict,
                  verbose=0)

        # 返回特征提取器
        feature_extractor = Model(inputs=model.input, outputs=model.get_layer('deep_features').output)
        return feature_extractor

    def _extract_cnn_features(self, X_scaled):
        """使用已训练的特征提取器提取深度特征"""
        n_features = X_scaled.shape[1]
        X_reshaped = X_scaled.reshape(-1, n_features, 1)
        return self.cnn_feature_extractor.predict(X_reshaped, verbose=0)

    def plot_importance(self, top_n=30):
        """绘制特征重要性（与原来类似）"""
        plt.figure(figsize=(6, 5))
        self.feature_importances_.head(top_n).sort_values().plot(kind='barh')
        plt.xlabel('Importance')
        plt.title('RandomForest Feature Importance (含深度特征)')
        plt.tight_layout()
        img_path = os.path.join(self.save_path, 'img', 'feature_importances.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

def split(file_path, target):
    # ====== 1. 数据预处理 ======
    preprocessor = CarClaimsPreprocessor(file_path)
    df = preprocessor.processor()   # 得到包含目标列的完整数据框

    # ====== 2. 划分数据集（先划分，再特征工程） ======
    # 先分出特征 X 和目标 y
    X = df.drop(columns=[target])
    y = df[target]

    # 划分训练、验证、测试集（例如 60% 训练，20% 验证，20% 测试）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # ====== 3. 特征选择（仅基于训练集拟合） ======
    fs = FeatureSelect(save_path='output/')
    # 在训练集上拟合特征选择器（包括 CNN 特征提取和随机森林）
    fs.fit(X_train, y_train,
           use_cnn=True,                # 根据需要开启
           cnn_output_dim=64,
           cnn_epochs=20,
           cnn_batch_size=32,
           rf_n_estimators=300)

    # 可视化特征重要性（可选）
    fs.plot_importance(top_n=30)

    # ====== 4. 转换所有数据集（应用同样的变换） ======
    X_train_trans = fs.transform(X_train)
    X_valid_trans = fs.transform(X_valid)
    X_test_trans  = fs.transform(X_test)

    # 此时 X_train_trans, X_valid_trans, X_test_trans 已经包含了深度特征（如果启用）

    return X_train_trans, y_train, X_valid_trans, y_valid, X_test_trans, y_test

# ========== 早停工具类 ==========
class EarlyStopping:
    """
    基于验证 AUC 的早停策略
    Args:
        patience (int): AUC 连续 patience 轮没有提升就停
        verbose (bool): 是否打印早停信息
        save_path (str): 最优模型保存路径
    """
    def __init__(self, patience=7, verbose=True, save_path=None):
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

class BaseModel:
    def __init__(self):
        self.model  = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_features = None  # 记录输入特征维度
        self.pos_weight = 1.0
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))

    @staticmethod
    def calc_pos_weight(y):
        """输入 0/1 向量，返回负/正比例 tensor"""
        y = np.asarray(y).squeeze()
        pos = y.sum()
        neg = len(y) - pos
        return torch.tensor(neg / max(pos, 1))   # 避免除 0
    
    def df_to_loader(self, X, y=None, shuffle=True):
        """
            将特征矩阵 X（及可选标签 y）打包成 PyTorch 的 DataLoader，供后续训练或预测使用。
        """

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

        # ---- 3. 组装成 DataLoader ----
        dataset = TensorDataset(X_tensor, y_tensor)
        loader  = DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=0)
        return loader
    
