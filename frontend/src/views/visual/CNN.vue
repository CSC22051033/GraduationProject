<template>
  <div class="card main-title">
    <h1>CNN 卷积神经网络模型</h1>
    <p class="subtitle">用于保险欺诈检测的深度学习模型</p>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在加载CNN模型结果...</p>
    </div>

    <!-- 错误信息 -->
    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <!-- 模型结果展示 -->
    <div v-if="!loading && !error" class="cnn-results">
      
      <!-- 第一行：训练历史图 -->
      <div class="row">
        <div class="image-card">
          <div class="image-header">
            <h3>CNN训练历史</h3>
            <p class="image-description">显示模型在训练过程中的准确率和损失变化</p>
          </div>
          <div class="image-container">
            <img 
              :src="trainingHistoryUrl" 
              alt="CNN训练历史" 
              class="cnn-image"
              @load="onTrainingHistoryLoad"
              @error="onTrainingHistoryError"
            />
            <div v-if="trainingHistoryLoading" class="image-loading">
              正在加载训练历史图...
            </div>
            <div v-if="trainingHistoryError" class="image-error">
              训练历史图加载失败
            </div>
          </div>
        </div>
      </div>

      <!-- 第二行：混淆矩阵和阈值优化 -->
      <div class="row">
        <div class="image-card half-width">
          <div class="image-header">
            <h3>混淆矩阵</h3>
            <p class="image-description">展示模型在测试集上的分类性能</p>
          </div>
          <div class="image-container">
            <img 
              :src="confusionMatrixUrl" 
              alt="CNN混淆矩阵" 
              class="cnn-image"
              @load="onConfusionMatrixLoad"
              @error="onConfusionMatrixError"
            />
            <div v-if="confusionMatrixLoading" class="image-loading">
              正在加载混淆矩阵...
            </div>
            <div v-if="confusionMatrixError" class="image-error">
              混淆矩阵加载失败
            </div>
          </div>
        </div>

        <div class="image-card half-width">
          <div class="image-header">
            <h3>阈值优化</h3>
            <p class="image-description">展示不同阈值下的模型性能指标</p>
          </div>
          <div class="image-container">
            <img 
              :src="thresholdOptimizationUrl" 
              alt="CNN阈值优化" 
              class="cnn-image"
              @load="onThresholdOptimizationLoad"
              @error="onThresholdOptimizationError"
            />
            <div v-if="thresholdOptimizationLoading" class="image-loading">
              正在加载阈值优化图...
            </div>
            <div v-if="thresholdOptimizationError" class="image-error">
              阈值优化图加载失败
            </div>
          </div>
        </div>
      </div>

      <!-- 模型信息 -->
      <div class="row">
        <div class="info-card">
          <h3>模型信息</h3>
          <div class="info-content">
            <!-- 第一行 -->
            <div class="info-row">
              <div class="info-item">
                <span class="info-label">模型类型:</span>
                <span class="info-value">卷积神经网络 (CNN)</span>
              </div>
              <div class="info-item">
                <span class="info-label">池化层:</span>
                <span class="info-value">全局最大池化</span>
              </div>
            </div>
            <!-- 第二行 -->
            <div class="info-row">
              <div class="info-item">
                <span class="info-label">卷积层:</span>
                <span class="info-value">2 层 (32, 64 滤波器)</span>
              </div>
              <div class="info-item">
                <span class="info-label">全连接层:</span>
                <span class="info-value">2 层 (64, 1 神经元)</span>
              </div>
            </div>
            <!-- 第三行 -->
            <div class="info-row">
              <div class="info-item">
                <span class="info-label">输出层:</span>
                <span class="info-value">1 神经元 (欺诈概率)</span>
              </div>
              <div class="info-item">
                <span class="info-label">保存位置:</span>
                <span class="info-value">output/models/cnn_model.pth</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 预测测试区域 -->
      <div class="row">
        <div class="prediction-card">
          <h3>模型测试</h3>
          <div class="prediction-controls">
            <!-- 增加输入框 -->
            <div class="input-group">
              <label for="test-index">测试样本索引 (k):</label>
              <input 
                type="number" 
                id="test-index"
                v-model.number="testIndex"
                min="0"
                placeholder="输入测试样本索引"
                class="index-input"
              />
            </div>
            
            <button 
              class="predict-btn"
              @click="testModel"
              :disabled="testing"
            >
              {{ testing ? '测试中...' : '运行模型测试' }}
            </button>
            
            <div v-if="testResult" class="test-result">
              <div class="result-header">
                <h4>测试结果</h4>
                <span class="sample-indicator">样本 #{{ testIndex }}</span>
              </div>
              <div class="result-content">
                <div class="result-row">
                  <div class="result-item">
                    <span class="result-label">欺诈概率</span>
                    <span class="result-value">{{ (testResult.fraud_probability * 100).toFixed(1) }}%</span>
                  </div>
                  <div class="result-item">
                    <span class="result-label">预测结果</span>
                    <span :class="['result-prediction', 
                                testResult.prediction === 'fraud' ? 'fraud' : 'non-fraud']">
                      {{ testResult.prediction === 'fraud' ? '欺诈' : '非欺诈' }}
                    </span>
                  </div>
                </div>
                <div class="result-row">
                  <div class="result-item">
                    <span class="result-label">真实标签</span>
                    <span :class="['result-prediction', 
                                testResult.true_value === 1 ? 'fraud' : 'non-fraud']">
                      {{ testResult.true_value === 1 ? '欺诈' : '非欺诈' }}
                    </span>
                  </div>
                  <div class="result-item">
                    <span class="result-label">是否匹配</span>
                    <span :class="['result-match', 
                                testResult.prediction === (testResult.true_value === 1 ? 'fraud' : 'non-fraud') ? 'match' : 'mismatch']">
                      {{ testResult.prediction === (testResult.true_value === 1 ? 'fraud' : 'non-fraud') ? '✓ 匹配' : '✗ 不匹配' }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: 'CNN',
  data() {
    return {
      loading: true,
      error: null,
      testing: false,
      testResult: null,
      testIndex: 0,  // 默认测试索引
      
      // 图片URL
      confusionMatrixUrl: 'http://localhost:5000/api/cnn_confusion_matrix',
      thresholdOptimizationUrl: 'http://localhost:5000/api/cnn_threshold_optimization',
      trainingHistoryUrl: 'http://localhost:5000/api/cnn_training_history',
      
      // 图片加载状态
      confusionMatrixLoading: true,
      thresholdOptimizationLoading: true,
      trainingHistoryLoading: true,
      
      confusionMatrixError: false,
      thresholdOptimizationError: false,
      trainingHistoryError: false
    };
  },
  mounted() {
    this.loadCNNResults();
  },
  methods: {
    async loadCNNResults() {
      this.loading = true;
      try {
        // 可以在这里添加其他数据加载逻辑
        // 比如加载模型评估指标等
        
        // 设置一个最小加载时间，避免闪烁
        setTimeout(() => {
          this.loading = false;
        }, 500);
        
      } catch (error) {
        console.error('加载CNN结果失败:', error);
        this.error = '加载CNN模型结果失败，请稍后重试';
        this.loading = false;
      }
    },
    
    async testModel() {
      this.testing = true;
      try {
        // 调用后端预测接口，传递测试索引
        const response = await fetch('http://localhost:5000/api/cnn_predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            k: this.testIndex  // 传递测试样本索引
          })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
          this.testResult = data.prediction;
        } else {
          throw new Error(data.error || '预测失败');
        }
      } catch (error) {
        console.error('模型测试失败:', error);
        alert(`模型测试失败: ${error.message}`);
      } finally {
        this.testing = false;
      }
    },
    
    // 图片加载成功处理
    onConfusionMatrixLoad() {
      this.confusionMatrixLoading = false;
    },
    onThresholdOptimizationLoad() {
      this.thresholdOptimizationLoading = false;
    },
    onTrainingHistoryLoad() {
      this.trainingHistoryLoading = false;
    },
    
    // 图片加载错误处理
    onConfusionMatrixError() {
      this.confusionMatrixLoading = false;
      this.confusionMatrixError = true;
      console.error('混淆矩阵图片加载失败');
    },
    onThresholdOptimizationError() {
      this.thresholdOptimizationLoading = false;
      this.thresholdOptimizationError = true;
      console.error('阈值优化图片加载失败');
    },
    onTrainingHistoryError() {
      this.trainingHistoryLoading = false;
      this.trainingHistoryError = true;
      console.error('训练历史图片加载失败');
    }
  }
};
</script>

<style scoped>
.main-title {
  padding: 20px;
}

.subtitle {
  color: #666;
  margin-bottom: 30px;
  font-size: 16px;
}

/* 加载状态 */
.loading {
  text-align: center;
  padding: 60px 20px;
  font-size: 18px;
  color: #666;
}

.loading .spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 错误信息 */
.error-message {
  background: #ffebee;
  color: #c62828;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 20px;
  border-left: 4px solid #f44336;
}

/* 行布局 */
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 20px;
}

/* 图片卡片 */
.image-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  overflow: hidden;
  flex: 1;
  min-width: 300px;
}

.half-width {
  flex: 0 0 calc(50% - 10px);
}

.image-header {
  padding: 15px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #eee;
}

.image-header h3 {
  margin: 0 0 5px 0;
  color: #333;
}

.image-description {
  margin: 0;
  font-size: 14px;
  color: #666;
}

.image-container {
  position: relative;
  padding: 20px;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cnn-image {
  max-width: 100%;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* 图片加载状态 */
.image-loading, .image-error {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 20px;
  background: rgba(255,255,255,0.9);
  border-radius: 4px;
  text-align: center;
  color: #666;
}

.image-error {
  color: #c62828;
  background: #ffebee;
}

/* 信息卡片 */
.info-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  padding: 20px;
  flex: 1;
}

.info-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.info-label {
  font-weight: 500;
  color: #555;
}

.info-value {
  color: #333;
  font-family: 'Courier New', monospace;
  background: #f5f5f5;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 14px;
}

/* 预测卡片 */
.prediction-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  padding: 20px;
  flex: 1;
}

.prediction-controls {
  margin-top: 20px;
}

.predict-btn {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}

.predict-btn:hover:not(:disabled) {
  background: #45a049;
}

.predict-btn:disabled {
  background: #cccccc;
  cursor: not-allowed;
}

.test-result {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
  border-left: 4px solid #4CAF50;
}

.test-result h4 {
  margin: 0 0 10px 0;
  color: #333;
}

.result-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.result-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.result-label {
  font-size: 14px;
  color: #666;
}

.result-value {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.result-prediction {
  padding: 4px 12px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 14px;
  display: inline-block;
}

.result-prediction.fraud {
  background: #ffebee;
  color: #c62828;
}

.result-prediction.non-fraud {
  background: #e8f5e8;
  color: #2e7d32;
}

/* 新增样式 */
/* 输入框样式优化 */
.input-group {
  margin-bottom: 20px;
}

.input-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.index-input {
  width: 97.5%;
  padding: 10px 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 15px;
  transition: all 0.3s ease;
  background-color: #fafafa;
}

.index-input:focus {
  outline: none;
  border-color: #007bff;
  background-color: white;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.index-input:hover {
  border-color: #b3b3b3;
}

/* 按钮样式优化 */
.predict-btn {
  width: 100%;
  padding: 12px 16px;
  background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
  font-size: 16px;
  transition: all 0.3s ease;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
}

.predict-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
  background: linear-gradient(135deg, #0069d9 0%, #004a99 100%);
}

.predict-btn:active:not(:disabled) {
  transform: translateY(0);
}

.predict-btn:disabled {
  background: #cccccc;
  cursor: not-allowed;
  box-shadow: none;
}

/* 结果区域样式优化 */
.test-result {
  margin-top: 20px;
  padding: 20px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 10px;
  border: 1px solid #dee2e6;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e9ecef;
}

.result-header h4 {
  margin: 0;
  color: #333;
  font-weight: 700;
}

.sample-indicator {
  background-color: #6c757d;
  color: white;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}

/* 结果内容样式优化 */
.result-row {
  display: flex;
  gap: 15px;
  margin-bottom: 15px;
}

.result-row:last-child {
  margin-bottom: 0;
}

.result-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.result-label {
  font-size: 13px;
  color: #666;
  font-weight: 500;
}

.result-value {
  font-size: 20px;
  font-weight: 700;
  color: #333;
}

/* 预测结果标签样式优化 */
.result-prediction {
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 14px;
  text-align: center;
  min-width: 80px;
  transition: all 0.3s ease;
}

.result-prediction.fraud {
  background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
  color: white;
  box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2);
}

.result-prediction.non-fraud {
  background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
  color: white;
  box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
}

/* 匹配结果样式优化 */
.result-match {
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 14px;
  text-align: center;
  min-width: 80px;
  transition: all 0.3s ease;
}

.result-match.match {
  background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
  color: white;
  box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
}

.result-match.mismatch {
  background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
  color: #333;
  box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .result-row {
    flex-direction: column;
    gap: 10px;
  }
  
  .result-item {
    width: 100%;
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .half-width {
    flex: 0 0 100%;
  }
  
  .info-content {
    grid-template-columns: 1fr;
  }
  
  .result-content {
    grid-template-columns: 1fr;
  }
}
</style>