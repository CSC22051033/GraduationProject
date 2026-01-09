<template>
  <div class="car-claims-visualization">
    <div class="header">
      <h1>汽车保险理赔数据集异常值处理</h1>
      <button @click="loadData" :disabled="loading" class="load-btn">
        {{ loading ? '加载中...' : '加载数据' }}
      </button>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="dataLoaded" class="content">
      <!-- 数据基本信息 -->
      <div class="info-section">
        <h2>数据基本信息</h2>
        <div class="info-grid">
          <div class="info-item">
            <label>数据形状:</label>
            <span>
              {{
                explorationData?.shape
                  ? `${explorationData.shape.rows} 行 × ${explorationData.shape.columns} 列`
                  : '加载中...'
              }}
            </span>
          </div>
        </div>
      </div>

      <!-- 异常特征信息 -->
      <div class="abnormal-section" v-if="abnormalFeatures && Object.keys(abnormalFeatures).length > 0">
        <h2>异常特征检测</h2>
        <div class="abnormal-features">
          <!-- 第一行：Age 和 DayOfWeekClaimed -->
          <div class="abnormal-row">
            <div v-if="abnormalFeatures.Age" class="abnormal-item">
              <h3>Age</h3>
              <p v-if="abnormalFeatures.Age.show_zero_info !== false">异常值(0)数量: {{ abnormalFeatures.Age.zero_count }}</p>
              <p v-if="abnormalFeatures.Age.show_zero_info !== false">异常值比例: {{ abnormalFeatures.Age.percentage }}%</p>
              <p>唯一值: {{ formatUniqueValues(abnormalFeatures.Age.unique_values) }}</p>
              <p>处理方式: {{ abnormalFeatures.Age.treatment }}</p>
            </div>
            <div v-if="abnormalFeatures.DayOfWeekClaimed" class="abnormal-item">
              <h3>DayOfWeekClaimed</h3>
              <p v-if="abnormalFeatures.DayOfWeekClaimed.show_zero_info !== false">异常值(0)数量: {{ abnormalFeatures.DayOfWeekClaimed.zero_count }}</p>
              <p v-if="abnormalFeatures.DayOfWeekClaimed.show_zero_info !== false">异常值比例: {{ abnormalFeatures.DayOfWeekClaimed.percentage }}%</p>
              <p>唯一值: {{ formatUniqueValues(abnormalFeatures.DayOfWeekClaimed.unique_values) }}</p>
              <p>处理方式: {{ abnormalFeatures.DayOfWeekClaimed.treatment }}</p>
            </div>
          </div>
          
          <!-- 第二行：Deductible 和 MonthClaimed -->
          <div class="abnormal-row">
            <div v-if="abnormalFeatures.Deductible" class="abnormal-item">
              <h3>Deductible</h3>
              <!-- Deductible不显示异常值数量和比例 -->
              <p>唯一值: {{ formatUniqueValues(abnormalFeatures.Deductible.unique_values) }}</p>
              <p>处理方式: {{ abnormalFeatures.Deductible.treatment }}</p>
            </div>
            <div v-if="abnormalFeatures.MonthClaimed" class="abnormal-item">
              <h3>MonthClaimed</h3>
              <p v-if="abnormalFeatures.MonthClaimed.show_zero_info !== false">异常值(0)数量: {{ abnormalFeatures.MonthClaimed.zero_count }}</p>
              <p v-if="abnormalFeatures.MonthClaimed.show_zero_info !== false">异常值比例: {{ abnormalFeatures.MonthClaimed.percentage }}%</p>
              <p>唯一值: {{ formatUniqueValues(abnormalFeatures.MonthClaimed.unique_values) }}</p>
              <p>处理方式: {{ abnormalFeatures.MonthClaimed.treatment }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 可视化控制 -->
      <div class="visualization-controls">
        <button @click="loadUniqueValuesPlot" class="plot-btn">
          显示唯一值分布
        </button>
        <button @click="loadAbnormalFeaturesPlot" class="plot-btn">
          显示异常值分布
        </button>
      </div>

      <!-- 可视化显示区域 -->
      <div class="visualization-section">
        <div v-if="currentPlot" class="plot-container">
          <h3>{{ getPlotTitle(currentPlot.plot_type) }}</h3>
          <img :src="currentPlot.image" :alt="currentPlot.plot_type" class="plot-image" />
        </div>
        
        <div v-else class="placeholder">
          <p>请选择上方的按钮来生成可视化图表</p>
        </div>
      </div>
    </div>

    <div v-else-if="!loading" class="welcome-message">
      <p>点击"加载数据"按钮开始分析汽车保险理赔数据</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

export default {
  name: 'DatasetException',
  data() {
    return {
      loading: false,
      dataLoaded: false,
      error: null,
      explorationData: {
        shape: null
      },
      abnormalFeatures: {},
      currentPlot: null,
      hasAutoLoaded: false // 添加标志防止重复加载
    };
  },
  mounted() {
    // 页面加载时自动调用 loadData
    this.loadData();
  },
  methods: {
    async loadData() {
      // 防止重复加载
      if (this.hasAutoLoaded) return;
      
      this.loading = true;
      this.error = null;
      try {
        const res = await axios.post(
          'http://localhost:5000/api/load_data',
          { file_path: 'carclaims.csv' },
          { withCredentials: true }
        );
        console.log('load_data 返回', res.data);
        if (res.data.success) {
          await this.loadExplorationData();
          this.dataLoaded = true;
          this.hasAutoLoaded = true; // 标记已加载
        } else {
          this.error = '数据加载失败: ' + (res.data.message || '未知错误');
        }
      } catch (err) {
        console.error('catch 到异常', err);
        this.error = `请求失败: ${err.message}`;
      } finally {
        this.loading = false;
      }
    },
    
    async loadExplorationData() {
      try {
        const res = await axios.get('http://localhost:5000/api/explore_data');
        console.log('explore_data 完整返回', res.data);
        this.explorationData = res.data;
        this.abnormalFeatures = res.data.abnormal_features || {};
      } catch (err) {
        console.error('explore_data 异常', err);
      }
    },
    
    async loadUniqueValuesPlot() {
      try {
        console.log("请求唯一值分布图...");
        const response = await axios.get(`${API_BASE}/unique_values_plot`);
        console.log("唯一值分布图响应:", response.data);
        
        if (response.data.image) {
          this.currentPlot = response.data;
          this.error = null;
        } else {
          this.error = '无法生成唯一值分布图: ' + (response.data.error || '未知错误');
          console.error('生成图表失败:', response.data);
        }
      } catch (err) {
        this.error = `获取唯一值分布图失败: ${err.message}`;
        console.error('获取唯一值分布图失败:', err);
        
        try {
          const debugResponse = await axios.get(`${API_BASE}/visualizer_debug`);
          console.log('可视化器调试信息:', debugResponse.data);
        } catch (debugErr) {
          console.error('获取调试信息失败:', debugErr);
        }
      }
    },

    async loadAbnormalFeaturesPlot() {
      try {
        console.log("请求异常值分布图...");
        const response = await axios.get(`${API_BASE}/abnormal_features_plot`);
        console.log("异常值分布图响应:", response.data);
        
        if (response.data.image) {
          this.currentPlot = response.data;
          this.error = null;
        } else {
          this.error = '无法生成异常值分布图: ' + (response.data.error || '未知错误');
          console.error('生成图表失败:', response.data);
        }
      } catch (err) {
        this.error = `获取异常值分布图失败: ${err.message}`;
        console.error('获取异常值分布图失败:', err);
        
        try {
          const debugResponse = await axios.get(`${API_BASE}/visualizer_debug`);
          console.log('可视化器调试信息:', debugResponse.data);
        } catch (debugErr) {
          console.error('获取调试信息失败:', debugErr);
        }
      }
    },
    
    getPlotTitle(plotType) {
      const titles = {
        'unique_values': 'MonthClaimed 和 DayOfWeekClaimed 唯一值分布',
        'abnormal_features': '异常特征分布图'
      };
      return titles[plotType] || '可视化图表';
    },

    formatUniqueValues(values) {
      if (!values || !Array.isArray(values)) return '';
      // 如果唯一值太多，只显示前5个
      if (values.length > 5) {
        return values.slice(0, 5).join(', ') + `... (共${values.length}个)`;
      }
      return values.join(', ');
    }
  }
};
</script>

<style scoped>
/* 样式保持不变，与之前相同 */
.car-claims-visualization {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #eee;
}

.load-btn, .plot-btn {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.load-btn:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.plot-btn {
  background-color: #2196F3;
  margin-right: 10px;
}

.error-message {
  color: #f44336;
  background-color: #ffebee;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.welcome-message {
  text-align: center;
  padding: 40px;
  color: #666;
}

.info-section, .abnormal-section {
  margin-bottom: 30px;
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 4px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.info-item {
  display: flex;
  flex-direction: column;
}

.info-item label {
  font-weight: bold;
  margin-bottom: 5px;
}

.abnormal-features {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.abnormal-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.abnormal-item {
  background-color: white;
  padding: 15px;
  border-radius: 4px;
  border-left: 4px solid #ff9800;
}

.abnormal-item h3 {
  margin-top: 0;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.abnormal-item p {
  margin: 8px 0;
  font-size: 14px;
  line-height: 1.4;
}

.visualization-controls {
  margin: 20px 0;
}

.visualization-section {
  margin-top: 20px;
}

.plot-container {
  text-align: center;
}

.plot-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.placeholder {
  text-align: center;
  padding: 40px;
  color: #999;
  background-color: #f5f5f5;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .abnormal-row {
    grid-template-columns: 1fr;
  }
  
  .header {
    flex-direction: column;
    gap: 15px;
    align-items: flex-start;
  }
}
</style>