<template>
  <div class="feature-container">
    <h1>数据集特征筛选</h1>
    
    <!-- 加载状态 -->
    <div v-if="loading" class="loading">
      正在加载数据...
    </div>
    
    <!-- 错误信息 -->
    <div v-if="error" class="error-message">
      {{ error }}
      <button @click="loadLatestFiles" class="retry-btn">重试</button>
    </div>
    
    <!-- 特征重要性图 -->
    <div v-if="!loading && !error" class="chart-section">
      <h2>前30个最重要特征</h2>
      <div class="chart-info" v-if="fileInfo.featureImportanceTime">
        生成时间: {{ fileInfo.featureImportanceTime }}
      </div>
      <div class="chart-container">
        <img 
          v-if="featureImportanceImage"
          :src="featureImportanceImage" 
          alt="特征重要性图" 
          class="chart-image"
          @load="onImageLoad"
          @error="onImageError"
        />
        <div v-else class="no-data">暂无特征重要性图片</div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'DatasetFeature',
  data() {
    return {
      featureImportanceImage: null,
      fileInfo: {
        featureImportanceTime: ''
      },
      loading: true,
      error: null,
      baseUrl: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000'
    };
  },
  mounted() {
    this.loadLatestFiles();
  },
  methods: {
    async loadLatestFiles() {
      this.loading = true;
      this.error = null;
      
      try {
        await this.loadFeatureImportanceImage();
      } catch (error) {
        console.error('加载图片失败:', error);
        this.error = '加载数据失败，请检查后端服务是否正常运行';
      } finally {
        this.loading = false;
      }
    },

    async loadFeatureImportanceImage() {
      try {
        const response = await axios.get(`${this.baseUrl}/api/feature_importances_image`, {
          responseType: 'blob'
        });
        
        if (response.status === 200) {
          const blob = new Blob([response.data], { type: 'image/png' });
          this.featureImportanceImage = URL.createObjectURL(blob);
          this.fileInfo.featureImportanceTime = new Date().toLocaleString('zh-CN');
          console.log('特征重要性图片加载成功');
        } else {
          throw new Error('无法获取特征重要性图片');
        }
      } catch (error) {
        console.error('加载特征重要性图片失败:', error);
        this.featureImportanceImage = null;
      }
    },

    onImageLoad() {
      console.log('特征重要性图片加载完成');
    },

    onImageError() {
      console.error('特征重要性图片加载失败');
      this.featureImportanceImage = null;
    }
  },
  
  beforeUnmount() {
    if (this.featureImportanceImage && this.featureImportanceImage.startsWith('blob:')) {
      URL.revokeObjectURL(this.featureImportanceImage);
    }
  }
}
</script>

<style scoped>
.feature-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

h1 {
  text-align: center;
  color: #333;
  margin-bottom: 30px;
}

h2 {
  color: #555;
  margin-bottom: 15px;
  border-bottom: 2px solid #e0e0e0;
  padding-bottom: 8px;
}

.loading {
  text-align: center;
  padding: 40px;
  font-size: 18px;
  color: #666;
}

.error-message {
  background: #ffebee;
  color: #c62828;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 20px;
  border-left: 4px solid #f44336;
  text-align: center;
}

.retry-btn {
  margin-top: 15px;
  padding: 8px 20px;
  background-color: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.retry-btn:hover {
  background-color: #1976d2;
}

.chart-section {
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.chart-info {
  font-size: 12px;
  color: #888;
  margin-bottom: 10px;
  font-style: italic;
}

.chart-container {
  text-align: center;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chart-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  display: block;
}

.no-data {
  text-align: center;
  color: #999;
  padding: 40px;
  font-style: italic;
  background: #f9f9f9;
  border-radius: 4px;
  width: 100%;
}

@media (max-width: 768px) {
  .feature-container {
    padding: 10px;
  }
  
  .chart-section {
    padding: 15px;
  }
}
</style>