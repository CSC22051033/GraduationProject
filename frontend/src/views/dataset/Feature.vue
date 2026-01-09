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
    </div>
    
    <!-- 第一行：图1和表1 -->
    <div v-if="!loading && !error" class="row-first">
      <!-- 图1：特征重要性图 -->
      <div class="chart-section">
        <h2>前20个最重要特征</h2>

      </div>
      
      <!-- 表1：特征列表 -->
      <div class="table-section">
        <h2>筛选出的所有特征</h2>
        <div class="table-info" v-if="fileInfo.featureListTime">
          筛选时间: {{ fileInfo.featureListTime }} | 特征数量: {{ fileInfo.featureCount }}
        </div>
        <div class="table-container">
          <table class="feature-table" v-if="features.length > 0">
            <thead>
              <tr>
                <th>序号</th>
                <th>特征</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(feature, index) in features" :key="index">
                <td class="index-column">{{ index + 1 }}</td>
                <td class="feature-column">{{ feature }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="no-data">未找到特征列表</div>
        </div>
      </div>
    </div>
    
    <!-- 第二行：图2 -->
    <div v-if="!loading && !error" class="row-second">
      <div class="chart-section full-width">
        <h2>训练历史</h2>
        
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
      trainingHistoryImage: null,
      features: [],
      fileInfo: {
        featureListTime: '',
        featureCount: 0,
        featureImportanceTime: '',
        trainingHistoryTime: ''
      },
      loading: true,
      error: null
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
        // 并行加载所有数据
        await Promise.all([
          // this.loadLatestFeatureImportanceImage(),
          // this.loadLatestTrainingHistoryImage(),
          this.loadLatestFeatureList()
        ]);
      } catch (error) {
        console.error('加载文件失败:', error);
        this.error = '加载数据失败，请检查后端服务是否正常运行';
      } finally {
        this.loading = false;
      }
    },

    async loadLatestFeatureImportanceImage() {
      try {
        // 首先尝试 base64 版本
        const response = await axios.get('http://localhost:5000/api/latest-feature-importance-image-base64');
        if (response.data && response.data.success && response.data.image) {
          this.featureImportanceImage = response.data.image;
          this.fileInfo.featureImportanceTime = this.formatTimestamp(response.data.timestamp);
          console.log('特征重要性图片加载成功 (base64)');
          return;
        }
      } catch (base64Error) {
        console.warn('Base64 版本加载失败，尝试 URL 版本:', base64Error);
        
        // 如果 base64 失败，尝试 URL 版本
        try {
          const urlResponse = await axios.get('http://localhost:5000/api/latest-feature-importance-image');
          if (urlResponse.data && urlResponse.data.success && urlResponse.data.url) {
            // 构建完整的图片 URL
            const fullImageUrl = `http://localhost:5000${urlResponse.data.url}`;
            this.featureImportanceImage = fullImageUrl;
            this.fileInfo.featureImportanceTime = this.formatTimestamp(urlResponse.data.timestamp);
            console.log('特征重要性图片加载成功 (URL):', fullImageUrl);
          }
        } catch (urlError) {
          console.error('URL 版本也加载失败:', urlError);
          throw new Error('无法加载特征重要性图片');
        }
      }
    },

    async loadLatestTrainingHistoryImage() {
      try {
        // 首先尝试 base64 版本
        const response = await axios.get('http://localhost:5000/api/latest-training-history-image-base64');
        if (response.data && response.data.success && response.data.image) {
          this.trainingHistoryImage = response.data.image;
          this.fileInfo.trainingHistoryTime = this.formatTimestamp(response.data.timestamp);
          console.log('训练历史图片加载成功 (base64)');
          return;
        }
      } catch (base64Error) {
        console.warn('Base64 版本加载失败，尝试 URL 版本:', base64Error);
        
        // 如果 base64 失败，尝试 URL 版本
        try {
          const urlResponse = await axios.get('http://localhost:5000/api/latest-training-history-image');
          if (urlResponse.data && urlResponse.data.success && urlResponse.data.url) {
            // 构建完整的图片 URL
            const fullImageUrl = `http://localhost:5000${urlResponse.data.url}`;
            this.trainingHistoryImage = fullImageUrl;
            this.fileInfo.trainingHistoryTime = this.formatTimestamp(urlResponse.data.timestamp);
            console.log('训练历史图片加载成功 (URL):', fullImageUrl);
          }
        } catch (urlError) {
          console.error('URL 版本也加载失败:', urlError);
          throw new Error('无法加载训练历史图片');
        }
      }
    },

    async loadLatestFeatureList() {
      try {
        const response = await axios.get('http://localhost:5000/api/latest-feature-list');
        if (response.data && response.data.success) {
          this.features = response.data.features || [];
          this.fileInfo.featureListTime = response.data.time || '';
          this.fileInfo.featureCount = response.data.count || 0;
          console.log('特征列表加载成功，数量:', this.features.length);
        } else {
          throw new Error(response.data?.error || '未知错误');
        }
      } catch (error) {
        console.error('加载特征列表失败:', error);
        throw new Error('无法加载特征列表');
      }
    },

    handleImageError(type) {
      console.error(`${type} 图片加载失败`);
      if (type === 'featureImportance') {
        this.featureImportanceImage = null;
      } else if (type === 'trainingHistory') {
        this.trainingHistoryImage = null;
      }
    },

    formatTimestamp(timestamp) {
      if (!timestamp) return '';
      const date = new Date(timestamp * 1000);
      return date.toLocaleString('zh-CN');
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
}

.row-first {
  display: flex;
  gap: 30px;
  margin-bottom: 30px;
}

.chart-section, .table-section {
  flex: 1;
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.full-width {
  flex: 1 1 100%;
}

.chart-info, .table-info {
  font-size: 12px;
  color: #888;
  margin-bottom: 10px;
  font-style: italic;
}

.chart-container {
  text-align: center;
}

.chart-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.table-container {
  max-height: 400px;
  overflow-y: auto;
}

.feature-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.feature-table th {
  background: #f5f5f5;
  padding: 12px 8px;
  text-align: left;
  font-weight: 600;
  color: #333;
  border-bottom: 2px solid #e0e0e0;
  position: sticky;
  top: 0;
}

.feature-table td {
  padding: 10px 8px;
  border-bottom: 1px solid #f0f0f0;
}

.feature-table tr:hover {
  background-color: #f8f9fa;
}

.index-column {
  width: 60px;
  text-align: center;
  color: #666;
  font-weight: 500;
}

.feature-column {
  word-break: break-word;
}

.no-data {
  text-align: center;
  color: #999;
  padding: 40px;
  font-style: italic;
  background: #f9f9f9;
  border-radius: 4px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .row-first {
    flex-direction: column;
  }
  
  .feature-container {
    padding: 10px;
  }
  
  .chart-section, .table-section {
    padding: 15px;
  }
}
</style>