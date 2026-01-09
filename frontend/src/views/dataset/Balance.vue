<template>
  <div class="balance-view">
    <h1>数据集平衡</h1>
    
    <div class="content">
      <!-- 加载状态 -->
      <div v-if="loading" class="loading">
        <p>加载平衡图表中...</p>
      </div>
      
      <!-- 错误信息 -->
      <div v-if="error" class="error-message">
        <p>{{ error }}</p>
        <button @click="loadBalanceImage" class="retry-btn">重试</button>
      </div>
      
      <!-- 图片显示区域 -->
      <div v-if="!loading && !error" class="image-section">
        <div class="image-container">
          <h2>类别分布平衡图 (SMOTE + 欠采样)</h2>
          <div class="image-wrapper">
            <img 
              :src="imageUrl" 
              alt="类别分布平衡图" 
              class="balance-image"
              @load="onImageLoad"
              @error="onImageError"
            />
          </div>
          <div class="image-description">
            <h3>平衡方法说明:</h3>
            <ul>
              <li><strong>SMOTE (Synthetic Minority Over-sampling Technique):</strong> 对少数类样本进行过采样，生成合成样本来平衡数据集。</li>
              <li><strong>欠采样 (Under-sampling):</strong> 对多数类样本进行欠采样，减少多数类样本数量。</li>
              <li>结合两种方法以达到更好的平衡效果。</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'DatasetBalance',
  data() {
    return {
      loading: true,
      error: null,
      imageUrl: '',
      imageLoaded: false,
      baseUrl: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000'
    };
  },
  mounted() {
    // 组件挂载时加载图片
    this.loadBalanceImage();
  },
  methods: {
    async loadBalanceImage() {
      this.loading = true;
      this.error = null;
      this.imageLoaded = false;
      
      try {
        // 方法1: 直接使用图片URL（需要后端配置静态文件服务）
        // this.imageUrl = `${this.baseUrl}/output/img/class_distribution_smote_under.png`;
        
        // 方法2: 通过API获取图片（更可靠）
        // 先测试API是否可用
        const testResponse = await axios.get(`${this.baseUrl}/api/balance_image`, {
          responseType: 'blob'
        });
        
        if (testResponse.status === 200) {
          // 创建对象URL
          const blob = new Blob([testResponse.data], { type: 'image/png' });
          this.imageUrl = URL.createObjectURL(blob);
        } else {
          throw new Error('无法获取图片');
        }
      } catch (err) {
        console.error('加载平衡图表失败:', err);
        
        // 如果API失败，尝试直接访问图片
        try {
          this.imageUrl = `${this.baseUrl}/output/img/class_distribution_smote_under.png`;
          
          // 测试图片是否可访问
          await axios.head(this.imageUrl);
        } catch (fallbackErr) {
          this.error = `无法加载平衡图表: ${err.message || '请检查后端服务是否运行'}`;
          
          // 提供一个占位符或错误图片
          this.imageUrl = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIzMDAiIHZpZXdCb3g9IjAgMCA4MDAgMzAwIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNmNWY1ZjUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjI0IiBmaWxsPSIjNjY2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Q2xhc3MgRGlzdHJpYnV0aW9uIENoYXJ0PC90ZXh0Pjx0ZXh0IHg9IjUwJSIgeT0iNjAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5TbW90ZSArIFVuZGVyLVNhbXBsaW5nIEJhbGFuY2luZzwvdGV4dD48L3N2Zz4=';
        }
      } finally {
        this.loading = false;
      }
    },
    
    onImageLoad() {
      this.imageLoaded = true;
      console.log('平衡图表加载成功');
    },
    
    onImageError() {
      this.error = '图片加载失败，请检查图片路径或网络连接';
      this.imageLoaded = false;
    }
  },
  // 修改前：beforeDestroy()
  // 修改后：使用 beforeUnmount（Vue 3）
  beforeUnmount() {
    // 组件销毁时释放对象URL
    if (this.imageUrl && this.imageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(this.imageUrl);
    }
  }
};
</script>

<style scoped>
.balance-view {
  padding: 20px;
  margin: 0 auto;
}

.balance-view h1 {
  margin-bottom: 30px;
  color: #333;
}

.content {
  background-color: #fff;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error-message {
  text-align: center;
  padding: 30px;
  background-color: #ffebee;
  border-radius: 4px;
  color: #d32f2f;
}

.retry-btn {
  margin-top: 15px;
  padding: 8px 20px;
  background-color: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.image-section {
  margin-top: 20px;
}

.image-container {
  text-align: center;
}

.image-container h2 {
  color: #333;
  margin: 10px auto 0;
  margin-bottom: 10px;
}

.image-wrapper {
  margin: 0 auto 30px;
  overflow: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
}

.balance-image {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

.image-description {
  text-align: left;
  max-width: 900px;
  margin: 5px auto 0;
  padding: 20px;
  border-radius: 4px;
}

.image-description h3 {
  color: #333;
  margin-bottom: 15px;
}

.image-description ul {
  padding-left: 20px;
}

.image-description li {
  margin-bottom: 10px;
  line-height: 1.6;
  color: #555;
}

@media (max-width: 768px) {
  .balance-view {
    padding: 10px;
  }
  
  .image-wrapper {
    max-width: 100%;
  }
}
</style>