<template>
  <div class="card main-title">
    <h1>LSTM 循环神经网络模型</h1>
    <p class="subtitle">用于保险欺诈检测的深度学习模型</p>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在加载LSTM模型结果...</p>
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
            <h3>LSTM训练历史</h3>
            <p class="image-description">显示模型在训练过程中的准确率和损失变化</p>
          </div>
          <div class="image-container">
            <img 
              :src="trainingHistoryUrl" 
              alt="LSTM训练历史" 
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

      <!-- 第二行：混淆矩阵和模型信息 -->
      <div class="row">
        <div class="image-card half-width">
          <div class="image-header">
            <h3>混淆矩阵</h3>
            <p class="image-description">展示模型在测试集上的分类性能</p>
          </div>
          <div class="image-container">
            <img 
              :src="confusionMatrixUrl" 
              alt="LSTM混淆矩阵" 
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

        <div class="info-card half-width">
          <h3>模型信息</h3>
          <div class="info-list">
            <div class="info-item">
              <span class="info-label">模型类型:</span>
              <span class="info-value">循环神经网络 (LSTM)</span>
            </div>
            <div class="info-item">
              <span class="info-label">批归一化:</span>
              <span class="info-value">2 层</span>
            </div>
            <div class="info-item">
              <span class="info-label">LSTM 层:</span>
              <span class="info-value">2 层 (128 隐藏单元)</span>
            </div>
            <div class="info-item">
              <span class="info-label">全连接层:</span>
              <span class="info-value">2 层 (64, 1 神经元)</span>
            </div>
            <div class="info-item">
              <span class="info-label">输出层:</span>
              <span class="info-value">1 神经元 (欺诈概率)</span>
            </div>
            <div class="info-item">
              <span class="info-label">保存位置:</span>
              <span class="info-value">output/models/rnn_model.pth</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: 'RNN',
  data() {
    return {
      loading: true,
      error: null,
      
      // 图片URL
      confusionMatrixUrl: 'http://localhost:5000/api/rnn_confusion_matrix',
      trainingHistoryUrl: 'http://localhost:5000/api/rnn_training_history',
      
      // 图片加载状态
      confusionMatrixLoading: true,
      trainingHistoryLoading: true,
      
      confusionMatrixError: false,
      trainingHistoryError: false
    };
  },
  mounted() {
    this.loadRNNResults();
  },
  methods: {
    async loadRNNResults() {
      this.loading = true;
      try {
        // 可以在这里添加其他数据加载逻辑
        // 比如加载模型评估指标等
        
        // 设置一个最小加载时间，避免闪烁
        setTimeout(() => {
          this.loading = false;
        }, 500);
        
      } catch (error) {
        console.error('加载RNN结果失败:', error);
        this.error = '加载RNN模型结果失败，请稍后重试';
        this.loading = false;
      }
    },
    
    // 图片加载成功处理
    onConfusionMatrixLoad() {
      this.confusionMatrixLoading = false;
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

.info-list {
  margin-top: 15px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
}

.info-item:last-child {
  border-bottom: none;
}

.info-label {
  font-weight: 600;
  color: #555;
  min-width: 120px;
  font-size: 14px;
}

.info-value {
  color: #333;
  font-family: 'Courier New', monospace;
  background: #f8f9fa;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 14px;
  flex: 1;
  text-align: right;
  border: 1px solid #e9ecef;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .half-width {
    flex: 0 0 100%;
  }
  
  .info-label {
    min-width: 100px;
    font-size: 13px;
  }
  
  .info-value {
    font-size: 13px;
  }
}
</style>