# flask 实现模型封装
import os
import glob
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from carClaims import CarClaimsPreprocessor  # 导入原有的类
from claims_visualizer import ClaimsVisualizer  # 导入新的可视化类
import base64
import io
import traceback
import re

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求
CORS(app, origins=['http://localhost:8080'], supports_credentials=True)

# 全局预处理器和可视化器实例
preprocessor = None
visualizer = None

from carClaims import split 
X_train, y_train, X_valid, y_valid, X_test, y_test = split('carclaims.csv', 'FraudFound')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """加载数据"""
    global preprocessor, visualizer
    
    try:
        data = request.get_json()
        file_path = data.get('file_path', 'carclaims.csv')
        
        preprocessor = CarClaimsPreprocessor(file_path)
        success = preprocessor.load_data()
        
        if success and preprocessor.df is not None: 
            visualizer = ClaimsVisualizer(preprocessor)
            return jsonify({
                "success": True,
                "message": "数据加载成功"
            })
        else:
            return jsonify({
                "success": False,
                "message": "数据加载失败: DataFrame 为空"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"数据加载错误: {str(e)}"
        })

@app.route('/api/explore_data', methods=['GET'])
def explore_data():
    """数据探索"""
    global preprocessor, visualizer
    
    if preprocessor is None or preprocessor.df is None:
        return jsonify({"error": "请先加载数据"}), 400
    
    try:
        # 获取数据形状
        data_shape = preprocessor.df.shape
        
        # 获取异常特征信息
        abnormal_features = {}
        if visualizer:
            try:
                abnormal_features = visualizer.get_abnormal_features()
            except Exception as e:
                print(f"获取异常特征时出错: {e}")
                traceback.print_exc()
                abnormal_features = {}
        
        # 手动添加Age和Deductible列的异常特征信息
        if preprocessor.df is not None:
            # 添加Age列的信息
            if 'Age' in preprocessor.df.columns:
                age_series = preprocessor.df['Age']
                zero_count = (age_series == 0).sum()
                total_count = len(age_series)
                zero_percentage = round((zero_count / total_count) * 100, 2) if total_count > 0 else 0
                
                # 计算除0外的int型平均数
                non_zero_ages = age_series[age_series != 0]
                if len(non_zero_ages) > 0:
                    avg_age = int(non_zero_ages.mean())
                else:
                    avg_age = 0
                
                abnormal_features['Age'] = {
                    'zero_count': int(zero_count),
                    'percentage': zero_percentage,
                    'unique_values': sorted(age_series.unique().tolist()),
                    'treatment': f'将异常值替换为除0外的int型平均数: {avg_age}',
                    'average_non_zero': avg_age,
                    'show_zero_info': True  # 标记是否显示异常值信息
                }
            
            # 添加Deductible列的信息
            if 'Deductible' in preprocessor.df.columns:
                deductible_series = preprocessor.df['Deductible']
                zero_count = (deductible_series == 0).sum()
                total_count = len(deductible_series)
                zero_percentage = round((zero_count / total_count) * 100, 2) if total_count > 0 else 0
                
                abnormal_features['Deductible'] = {
                    'zero_count': int(zero_count),
                    'percentage': zero_percentage,
                    'unique_values': sorted(deductible_series.unique().tolist()),
                    'treatment': '为防止信息丢失，不做异常处理',
                    'preserve_data': True,
                    'show_zero_info': False  # 标记不显示异常值信息
                }
            
            # 为DayOfWeekClaimed和MonthClaimed添加处理方式
            if 'DayOfWeekClaimed' in abnormal_features:
                abnormal_features['DayOfWeekClaimed']['treatment'] = '将异常值替换为众数'
                abnormal_features['DayOfWeekClaimed']['show_zero_info'] = True
            
            if 'MonthClaimed' in abnormal_features:
                abnormal_features['MonthClaimed']['treatment'] = '将异常值替换为众数'
                abnormal_features['MonthClaimed']['show_zero_info'] = True
        
        return jsonify({
            "shape": {
                "rows": data_shape[0],
                "columns": data_shape[1]
            },
            "abnormal_features": abnormal_features
        })
    except Exception as e:
        print(f"数据探索失败: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"数据探索失败: {str(e)}"
        }), 500


@app.route('/api/unique_values_plot', methods=['GET'])
def get_unique_values_plot():
    """获取唯一值数量可视化图"""
    global visualizer
    
    if visualizer is None:
        print("错误: visualizer 为 None")
        return jsonify({"error": "请先加载数据"})
    
    try:
        # 检查 ClaimsVisualizer 是否有 plot_unique_values 方法
        if hasattr(visualizer, 'plot_unique_values'):
            print("开始生成唯一值图表...")
            plot_buffer = visualizer.plot_unique_values()
            
            if plot_buffer:
                print("图表生成成功，开始编码...")
                # 将图像转换为base64
                plot_buffer.seek(0)
                plot_data = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
                plot_buffer.close()
                print("图表编码完成")
                return jsonify({
                    "plot_type": "unique_values",
                    "image": f"data:image/png;base64,{plot_data}"
                })
            else:
                print("plot_unique_values 返回了 None")
                return jsonify({"error": "图表生成为空"})
        else:
            print(f"visualizer 没有 plot_unique_values 方法，可用方法: {[m for m in dir(visualizer) if not m.startswith('_')]}")
            return jsonify({"error": "可视化方法不可用"})
    except Exception as e:
        print(f"生成唯一值图表时出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"生成图表失败: {str(e)}"
        })

@app.route('/api/abnormal_features_plot', methods=['GET'])
def get_abnormal_features_plot():
    """获取异常特征可视化图"""
    global visualizer
    
    if visualizer is None:
        print("错误: visualizer 为 None")
        return jsonify({"error": "请先加载数据"})
    
    try:
        # 检查 ClaimsVisualizer 是否有 plot_abnormal_features 方法
        if hasattr(visualizer, 'plot_abnormal_features'):
            print("开始生成异常特征图表...")
            plot_buffer = visualizer.plot_abnormal_features()
            
            if plot_buffer:
                print("异常特征图表生成成功，开始编码...")
                # 将图像转换为base64
                plot_buffer.seek(0)
                plot_data = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
                plot_buffer.close()
                print("异常特征图表编码完成")
                return jsonify({
                    "plot_type": "abnormal_features",
                    "image": f"data:image/png;base64,{plot_data}"
                })
            else:
                print("plot_abnormal_features 返回了 None")
                return jsonify({"error": "异常特征图表生成为空"})
        else:
            print(f"visualizer 没有 plot_abnormal_features 方法，可用方法: {[m for m in dir(visualizer) if not m.startswith('_')]}")
            return jsonify({"error": "可视化方法不可用"})
    except Exception as e:
        print(f"生成异常特征图表时出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"生成图表失败: {str(e)}"
        })

@app.route('/api/abnormal_features_info', methods=['GET'])
def get_abnormal_features_info():
    """获取异常特征详细信息"""
    global visualizer
    
    if visualizer is None:
        return jsonify({"error": "请先加载数据"})
    
    try:
        abnormal_info = visualizer.get_abnormal_features()
        return jsonify({
            "abnormal_features": abnormal_info
        })
    except Exception as e:
        return jsonify({
            "error": f"获取异常特征信息失败: {str(e)}"
        })

@app.route('/api/data_info', methods=['GET'])
def get_data_info():
    """获取数据基本信息"""
    global preprocessor
    
    if preprocessor is None or preprocessor.df is None:
        return jsonify({"error": "请先加载数据"})
    
    try:
        df = preprocessor.df
        return jsonify({
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "memory_usage": df.memory_usage(deep=True).sum()
        })
    except Exception as e:
        return jsonify({
            "error": f"获取数据信息失败: {str(e)}"
        })

@app.route('/api/detect_outliers', methods=['GET'])
def detect_outliers():
    """使用原有的异常值检测方法"""
    global preprocessor
    
    if preprocessor is None or preprocessor.df is None:
        return jsonify({"error": "请先加载数据"})
    
    try:
        # 调用原有的_preliminary_outlier_detection方法
        preprocessor._preliminary_outlier_detection()
        return jsonify({
            "success": True,
            "message": "异常值检测完成"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy"})

@app.route('/api/latest-feature-importance-image', methods=['GET'])
def get_latest_feature_importance_image():
    """获取最新的特征重要性图片"""
    try:
        img_dir = 'output/img'
        print(f"查找图片目录: {img_dir}")
        
        if not os.path.exists(img_dir):
            print(f"图片目录不存在: {img_dir}")
            return jsonify({"error": "图片目录不存在"}), 404
        
        # 查找所有特征重要性图片文件
        pattern = os.path.join(img_dir, 'feature_importance_*.png')
        files = glob.glob(pattern)
        print(f"找到的特征重要性图片文件: {files}")
        
        if not files:
            return jsonify({"error": "未找到特征重要性图片"}), 404
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)
        print(f"选择的特征重要性图片: {latest_file}")
        
        # 返回文件路径供前端使用
        return jsonify({
            "success": True,
            "url": f"/api/image/feature_importance/{os.path.basename(latest_file)}",
            "filename": os.path.basename(latest_file),
            "timestamp": os.path.getmtime(latest_file)
        })
        
    except Exception as e:
        print(f"获取特征重要性图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取特征重要性图片失败: {str(e)}"}), 500

@app.route('/api/latest-training-history-image', methods=['GET'])
def get_latest_training_history_image():
    """获取最新的训练历史图片"""
    try:
        img_dir = 'output/img'
        print(f"查找训练历史图片目录: {img_dir}")
        
        if not os.path.exists(img_dir):
            return jsonify({"error": "图片目录不存在"}), 404
        
        # 查找所有训练历史图片文件
        pattern = os.path.join(img_dir, 'training_history_*.png')
        files = glob.glob(pattern)
        print(f"找到的训练历史图片文件: {files}")
        
        if not files:
            return jsonify({"error": "未找到训练历史图片"}), 404
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)
        print(f"选择的训练历史图片: {latest_file}")
        
        # 返回文件路径供前端使用
        return jsonify({
            "success": True,
            "url": f"/api/image/training_history/{os.path.basename(latest_file)}",
            "filename": os.path.basename(latest_file),
            "timestamp": os.path.getmtime(latest_file)
        })
        
    except Exception as e:
        print(f"获取训练历史图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取训练历史图片失败: {str(e)}"}), 500

@app.route('/api/image/feature_importance/<filename>', methods=['GET'])
def serve_feature_importance_image(filename):
    """提供特征重要性图片文件"""
    try:
        file_path = os.path.join('output/img', filename)
        print(f"尝试提供特征重要性图片: {file_path}")
        
        if not os.path.exists(file_path):
            return jsonify({"error": "图片文件不存在"}), 404
        
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        print(f"提供特征重要性图片失败: {str(e)}")
        return jsonify({"error": f"提供图片失败: {str(e)}"}), 500

@app.route('/api/image/training_history/<filename>', methods=['GET'])
def serve_training_history_image(filename):
    """提供训练历史图片文件"""
    try:
        file_path = os.path.join('output/img', filename)
        print(f"尝试提供训练历史图片: {file_path}")
        
        if not os.path.exists(file_path):
            return jsonify({"error": "图片文件不存在"}), 404
        
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        print(f"提供训练历史图片失败: {str(e)}")
        return jsonify({"error": f"提供图片失败: {str(e)}"}), 500

@app.route('/api/latest-feature-list', methods=['GET'])
def get_latest_feature_list():
    """获取最新的特征列表"""
    try:
        feature_lists_dir = 'output/feature_lists'
        print(f"查找特征列表目录: {feature_lists_dir}")
        
        if not os.path.exists(feature_lists_dir):
            return jsonify({"error": "特征列表目录不存在"}), 404
        
        # 查找所有特征列表文件
        pattern = os.path.join(feature_lists_dir, 'selected_features_simple*.txt')
        files = glob.glob(pattern)
        print(f"找到的特征列表文件: {files}")
        
        if not files:
            return jsonify({"error": "未找到特征列表文件"}), 404
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)
        print(f"选择的特征列表文件: {latest_file}")
        
        # 读取文件内容
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析文件内容
        features = []
        time = ""
        count = 0
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # 使用正则表达式匹配格式： " 1. BasePolicy_Liability"
            match = re.match(r'^\s*(\d+)\.\s*(.+)$', line)
            if match:
                features.append(match.group(2))
            # 解析文件头信息
            elif line.startswith('筛选时间:'):
                time = line.replace('筛选时间:', '').strip()
            elif line.startswith('特征数量:'):
                count_str = line.replace('特征数量:', '').strip()
                count = int(count_str) if count_str.isdigit() else 0
        
        print(f"解析出的特征数量: {len(features)}")
        
        return jsonify({
            "success": True,
            "features": features,
            "time": time,
            "count": count,
            "filename": os.path.basename(latest_file),
            "timestamp": os.path.getmtime(latest_file)
        })
        
    except Exception as e:
        print(f"获取特征列表失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取特征列表失败: {str(e)}"}), 500

@app.route('/api/latest-feature-importance-image-base64', methods=['GET'])
def get_latest_feature_importance_image_base64():
    """获取最新的特征重要性图片（base64编码）"""
    try:
        img_dir = 'output/img'
        print(f"查找特征重要性图片目录 (base64): {img_dir}")
        
        if not os.path.exists(img_dir):
            return jsonify({"error": "图片目录不存在"}), 404
        
        # 查找所有特征重要性图片文件
        pattern = os.path.join(img_dir, 'feature_importance_*.png')
        files = glob.glob(pattern)
        print(f"找到的特征重要性图片文件 (base64): {files}")
        
        if not files:
            return jsonify({"error": "未找到特征重要性图片"}), 404
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)
        print(f"选择的特征重要性图片 (base64): {latest_file}")
        
        # 读取图片并转换为base64
        with open(latest_file, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_data}",
            "filename": os.path.basename(latest_file),
            "timestamp": os.path.getmtime(latest_file)
        })
        
    except Exception as e:
        print(f"获取特征重要性图片失败 (base64): {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取特征重要性图片失败: {str(e)}"}), 500

@app.route('/api/latest-training-history-image-base64', methods=['GET'])
def get_latest_training_history_image_base64():
    """获取最新的训练历史图片（base64编码）"""
    try:
        img_dir = 'output/img'
        print(f"查找训练历史图片目录 (base64): {img_dir}")
        
        if not os.path.exists(img_dir):
            return jsonify({"error": "图片目录不存在"}), 404
        
        # 查找所有训练历史图片文件
        pattern = os.path.join(img_dir, 'training_history_*.png')
        files = glob.glob(pattern)
        print(f"找到的训练历史图片文件 (base64): {files}")
        
        if not files:
            return jsonify({"error": "未找到训练历史图片"}), 404
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)
        print(f"选择的训练历史图片 (base64): {latest_file}")
        
        # 读取图片并转换为base64
        with open(latest_file, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_data}",
            "filename": os.path.basename(latest_file),
            "timestamp": os.path.getmtime(latest_file)
        })
        
    except Exception as e:
        print(f"获取训练历史图片失败 (base64): {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取训练历史图片失败: {str(e)}"}), 500

@app.route('/api/balance_image')
def get_balance_image():
    try:
        # 图片路径，根据你的项目结构调整
        image_path = os.path.join('..', 'output', 'img', 'class_distribution_smote_under.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# CNN
@app.route('/api/cnn_confusion_matrix')
def get_cnn_confusion_matrix():
    """获取CNN混淆矩阵图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'cnn_confusion_matrix.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/cnn_threshold_optimization')
def get_cnn_threshold_optimization():
    """获取CNN阈值优化图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'cnn_threshold_optimization.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/cnn_training_history')
def get_cnn_training_history():
    """获取CNN训练历史图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'cnn_training_history.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/cnn_predict', methods=['POST'])
def cnn_predict():
    """CNN模型预测接口"""
    try:
        from DeepLearning_CNN import CNNModel

        data = request.get_json()
        # 获取前端传递的测试样本索引k，默认为0
        k = data.get('k', 0) if data else 0
        
        cnn_model = CNNModel()
        try:
            cnn_model.load()
        except:
            return jsonify({
                'error': '模型未训练，请先训练模型',
                'status': 'failed'
            }), 400
        
        prob, pred = cnn_model.predict(
            x_single=X_test.iloc[k],
            best_threshold=0.5
        )

        prediction = {
            "fraud_probability": float(prob),
            "prediction": "fraud" if pred == 1 else "non-fraud",
            "true_value": int(y_test.iloc[k])
        }

        return jsonify({
            'status': 'success',
            'prediction': prediction
        }), 200

    except Exception as e:
        print(f"CNN预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"预测失败: {str(e)}"
        }), 500

# RNN
@app.route('/api/rnn_confusion_matrix')
def get_rnn_confusion_matrix():
    """获取RNN混淆矩阵图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'rnn_confusion_matrix.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/rnn_threshold_optimization')
def get_rnn_threshold_optimization():
    """获取RNN阈值优化图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'rnn_threshold_optimization.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/rnn_training_history')
def get_rnn_training_history():
    """获取RNN训练历史图片"""
    try:
        image_path = os.path.join('..', 'output', 'img', 'rnn_training_history.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404
    
@app.route('/api/rnn_predict', methods=['POST'])
def rnn_predict():
    """RNN模型预测接口"""
    try:
        from DeepLearning_RNN import RNNModel

        data = request.get_json()
        # 获取前端传递的测试样本索引k，默认为0
        k = data.get('k', 0) if data else 0
        
        rnn_model = RNNModel()
        try:
            rnn_model.load('output/model/rnn_model.pth')
        except:
            return jsonify({
                'error': '模型未训练，请先训练模型',
                'status': 'failed'
            }), 400
        
        best_threshold = rnn_model.optimize_threshold(X_valid, y_valid)

        prob, pred = rnn_model.predict(
            x_single=X_test.iloc[k],
            best_threshold=best_threshold
        )

        prediction = {
            "fraud_probability": float(prob),
            "prediction": "fraud" if pred == 1 else "non-fraud",
            "true_value": int(y_test.iloc[k])
        }

        return jsonify({
            'status': 'success',
            'prediction': prediction
        }), 200

    except Exception as e:
        print(f"RNN预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"预测失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)