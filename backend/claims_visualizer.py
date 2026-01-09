import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

# 设置字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

class ClaimsVisualizer:
    def __init__(self, preprocessor):
        """
        初始化可视化器
        Args:
            preprocessor: CarClaimsPreprocessor实例
        """
        self.preprocessor = preprocessor
        self.df = preprocessor.df

    def get_abnormal_features(self):
        """获取存在异常值的特征信息"""
        if self.df is None:
            return {"error": "请先加载数据！"}
        
        abnormal_features = {}
        
        # 检查 MonthClaimed 和 DayOfWeekClaimed 的异常值
        for col in ['MonthClaimed', 'DayOfWeekClaimed']:
            if col in self.df.columns:
                # 检查数值型和字符串型的0值
                if self.df[col].dtype in ['int64', 'float64']:
                    zero_count = (self.df[col] == 0).sum()
                else:
                    zero_count = (self.df[col].astype(str) == '0').sum()
                
                if zero_count > 0:
                    abnormal_features[col] = {
                        "zero_count": int(zero_count),
                        "total_count": len(self.df),
                        "percentage": round((zero_count / len(self.df)) * 100, 2),
                        "unique_values": self.df[col].unique().tolist()
                    }
        
        return abnormal_features

    def plot_unique_values(self):
        """生成唯一值数量可视化图"""
        if self.df is None:
            return None
        
        # 创建图形
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MonthClaimed 的唯一值分布
        if 'MonthClaimed' in self.df.columns:
            month_data = self.df['MonthClaimed']
            # 处理不同类型的数据
            if month_data.dtype in ['int64', 'float64']:
                month_counts = month_data.value_counts().sort_index()
            else:
                month_counts = month_data.astype(str).value_counts().sort_index()
            
            bars1 = ax1.bar([str(x) for x in month_counts.index], month_counts.values)
            ax1.set_title('MonthClaimed 唯一值分布')
            ax1.set_xlabel('月份')
            ax1.set_ylabel('频次')
            ax1.tick_params(axis='x', rotation=45)
            
            # 标记异常值（值为0）
            if 0 in month_counts.index or '0' in month_counts.index:
                zero_key = 0 if 0 in month_counts.index else '0'
                zero_idx = list(month_counts.index).index(zero_key)
                bars1[zero_idx].set_color('red')
                ax1.text(zero_idx, month_counts[zero_key], '异常值', 
                        ha='center', va='bottom', color='red', fontweight='bold')
        
        # DayOfWeekClaimed 的唯一值分布
        if 'DayOfWeekClaimed' in self.df.columns:
            day_data = self.df['DayOfWeekClaimed']
            # 处理不同类型的数据
            if day_data.dtype in ['int64', 'float64']:
                day_counts = day_data.value_counts().sort_index()
            else:
                day_counts = day_data.astype(str).value_counts().sort_index()
            
            bars2 = ax2.bar([str(x) for x in day_counts.index], day_counts.values)
            ax2.set_title('DayOfWeekClaimed 唯一值分布')
            ax2.set_xlabel('星期几')
            ax2.set_ylabel('频次')
            ax2.tick_params(axis='x', rotation=45)
            
            # 标记异常值（值为0）
            if 0 in day_counts.index or '0' in day_counts.index:
                zero_key = 0 if 0 in day_counts.index else '0'
                zero_idx = list(day_counts.index).index(zero_key)
                bars2[zero_idx].set_color('red')
                ax2.text(zero_idx, day_counts[zero_key], '异常值', 
                        ha='center', va='bottom', color='red', fontweight='bold')
        
        plt.tight_layout()
        
        # 转换为BytesIO对象（不进行base64编码）
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer  # 返回BytesIO对象，不是base64字符串

    def plot_abnormal_features(self):
        """
        使用IQR方法检测数值型列的异常值并绘制可视化图
        Returns:
            BytesIO缓冲区对象，如果没有异常特征则返回None
        """
        try:
            # 获取数值型列
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                return None
            
            # 检测异常特征
            abnormal_features = self._detect_abnormal_features(numeric_columns)
            
            if not abnormal_features:
                return None
            
            # 创建可视化图表
            return self._create_abnormal_plot(abnormal_features, numeric_columns)
            
        except Exception as e:
            print(f"绘制异常特征图时出错: {str(e)}")
            return None

    def _create_abnormal_plot(self, abnormal_features, numeric_columns):
        """
        创建异常特征可视化图
        """
        # 设置图形大小 - 2行2列的子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数值型特征异常值检测分析', fontsize=16, fontweight='bold')
        
        # 子图1：箱线图展示异常值分布 - 使用第一个子图 axes[0, 0]
        self._plot_boxplots(axes[0, 0], numeric_columns)
        
        # 子图2：异常值统计摘要 - 使用第二个子图 axes[0, 1]
        self._plot_summary_table(axes[0, 1], abnormal_features)
        
        # 子图3：特征Age的唯一值分布 - 使用第三个子图 axes[1, 0]
        self._plot_unique_value_distribution(axes[1, 0], 'Age')
        
        # 子图4：特征Deductible的唯一值分布 - 使用第四个子图 axes[1, 1]
        self._plot_unique_value_distribution(axes[1, 1], 'Deductible')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
        
        # 返回缓冲区对象而不是base64字符串
        return self._fig_to_buffer(fig)

    def _plot_unique_value_distribution(self, ax, column_name):
        """
        绘制指定列的唯一值分布
        
        Args:
            ax: 子图轴对象
            column_name: 列名
        """
        try:
            # 检查列是否存在
            if column_name not in self.df.columns:
                ax.text(0.5, 0.5, f"列 '{column_name}' 不存在", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{column_name} - 唯一值分布')
                return
            
            # 获取列数据
            column_data = self.df[column_name].dropna()
            
            if len(column_data) == 0:
                ax.text(0.5, 0.5, f"列 '{column_name}' 无数据", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{column_name} - 唯一值分布')
                return
            
            # 计算唯一值的计数
            value_counts = column_data.value_counts().sort_index()
            
            ax.set_title(f'{column_name} - 唯一值分布')
            
            # 创建条形图
            bars = ax.bar(range(len(value_counts)), value_counts.values)
            
            # 设置x轴标签
            ax.set_xticks(range(len(value_counts)))
            
            # 判断是否需要简化显示（当唯一值过多时）
            too_many_values = len(value_counts) > 15
            
            if too_many_values:
                # 唯一值过多时，不显示所有x轴标签
                # 只显示部分标签（例如每5个显示一个）
                step = max(1, len(value_counts) // 10)  # 大约显示10个标签
                visible_indices = list(range(0, len(value_counts), step))
                ax.set_xticks(visible_indices)
                ax.set_xticklabels([value_counts.index[i] for i in visible_indices], rotation=45, ha='right')
            else:
                # 唯一值不多时，正常显示所有标签
                if (value_counts.index.dtype.kind in 'biufc' and 
                    (value_counts.index.max() - value_counts.index.min()) > 100) or \
                any(len(str(x)) > 5 for x in value_counts.index):
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                else:
                    ax.set_xticklabels(value_counts.index)
            
            ax.set_xlabel('值')
            ax.set_ylabel('频数')
            
            # 只有当唯一值不多时才在条形上添加数值标签
            if not too_many_values:
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            # 添加网格线
            ax.grid(True, axis='y', alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"绘制错误: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{column_name} - 唯一值分布')

    def _fig_to_buffer(self, fig):
        """将matplotlib图形转换为BytesIO缓冲区"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # 关闭图形释放内存
        return buf  # 返回缓冲区对象
        
    def _detect_abnormal_features(self, numeric_columns):
        """
        使用IQR方法检测异常特征
        """
        abnormal_features = {}
        
        for col in numeric_columns:
            # 移除NaN值
            data = self.df[col].dropna()
            
            if len(data) < 4:  # 数据太少无法计算IQR
                continue
                
            # 计算IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # 异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 检测异常值
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            if len(outliers) > 0:
                abnormal_features[col] = {
                    'total_count': len(data),
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers': outliers.tolist()
                }
        
        return abnormal_features

    def _plot_boxplots(self, ax, numeric_columns):
        """绘制箱线图"""
        # 使用所有数值特征进行箱线图展示
        plot_columns = numeric_columns
        plot_data = self.df[plot_columns]
        
        # 对数据进行归一化处理（最小-最大归一化）
        normalized_data = []
        for col in plot_columns:
            col_data = plot_data[col].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                # 避免除以零
                if max_val != min_val:
                    normalized_col = (col_data - min_val) / (max_val - min_val)
                else:
                    normalized_col = col_data * 0  # 所有值相同，归一化后为0
                normalized_data.append(normalized_col)
            else:
                normalized_data.append(pd.Series([]))  # 空数据
        
        # 创建归一化后的箱线图
        boxplot = ax.boxplot(normalized_data, 
                        labels=plot_columns, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue' for _ in plot_columns]
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('数值特征归一化分布箱线图')
        ax.set_xlabel('特征')
        ax.set_ylabel('归一化数值')
        ax.tick_params(axis='x', rotation=90)  # 旋转90度以避免标签重叠
        
        # 如果特征过多，调整布局
        if len(plot_columns) > 10:
            # 调整底部边距以容纳更多标签
            plt.subplots_adjust(bottom=0.3)

    def _plot_summary_table(self, ax, abnormal_features):
        """绘制统计摘要表格"""
        if not abnormal_features:
            ax.text(0.5, 0.5, '未检测到异常特征', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('异常特征统计摘要')
            return
        
        # 准备表格数据
        table_data = []
        for feat, stats in abnormal_features.items():
            table_data.append([
                feat,
                stats['total_count'],
                stats['outlier_count'],
                f"{stats['outlier_percentage']:.2f}%",
                f"{stats['lower_bound']:.2f}",
                f"{stats['upper_bound']:.2f}"
            ])
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['特征', '总数', '异常数', '异常比例', '下界', '上界'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.1, 0.9, 0.8])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置标题行样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('异常特征统计摘要')
        ax.axis('off')


    
    