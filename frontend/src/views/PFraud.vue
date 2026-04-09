<template>
  <section class="pfraud-page">
    <div class="page-title">
      <h1>欺诈概率输出</h1>
      <p>加载测试集并使用模型对100条数据进行预测，展示预测结果与实际值对比。</p>
    </div>

    <div class="actions-panel">
      <button class="primary-button" @click="loadTestData" :disabled="loadingData">
        {{ loadingData ? '正在加载测试集...' : '加载测试集' }}
      </button>

      <div class="select-group">
        <label for="model-select">选择模型</label>
        <select id="model-select" v-model="selectedModel">
          <option value="cnn">CNN</option>
          <option value="rnn">LSTM</option>
          <option value="mix">混合模型</option>
        </select>
      </div>

      <button class="secondary-button" @click="predictModel" :disabled="loadingPrediction || !testSamples.length">
        {{ loadingPrediction ? '正在预测...' : '开始预测' }}
      </button>
    </div>

    <div v-if="error" class="error-box">
      <strong>错误：</strong> {{ error }}
    </div>

    <div v-if="testSamples.length" class="data-preview card">
      <div class="card-header">
        <h2>测试集预览</h2>
        <span>已加载 {{ testSamples.length }} 条测试数据</span>
      </div>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th v-for="column in previewColumns" :key="column">{{ column }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, index) in testSamples.slice(0, 5)" :key="index">
              <td>{{ index + 1 }}</td>
              <td v-for="column in previewColumns" :key="column">{{ row[column] }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="note">仅显示前 5 条记录。如需对比完整预测结果，请点击“开始预测”。</p>
    </div>

    <div v-if="predictions.length" class="result-card card">
      <div class="card-header">
        <h2>预测结果对比</h2>
        <span>模型：{{ modelLabel }}</span>
      </div>
      <div class="result-summary">
        <div>共预测 {{ predictions.length }} 条数据</div>
        <div>预测欺诈数：{{ fraudCount }} / 实际欺诈数：{{ actualFraudCount }}</div>
      </div>
      <div class="table-wrapper">
        <table class="result-table">
          <thead>
            <tr>
              <th>#</th>
              <th>欺诈概率</th>
              <th>模型预测</th>
              <th>真实值</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in predictions" :key="row.index">
              <td>{{ row.index + 1 }}</td>
              <td>{{ row.fraud_probability.toFixed(4) }}</td>
              <td>{{ row.prediction }}</td>
              <td>{{ row.true_value === 1 ? 'fraud' : 'non-fraud' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'PFraud',
  data() {
    return {
      loadingData: false,
      loadingPrediction: false,
      error: null,
      selectedModel: 'cnn',
      testSamples: [],
      predictions: []
    };
  },
  computed: {
    previewColumns() {
      if (!this.testSamples.length) {
        return [];
      }
      return Object.keys(this.testSamples[0]).filter(key => key !== 'true_value');
    },
    modelLabel() {
      return {
        cnn: 'CNN',
        rnn: 'LSTM',
        mix: '混合模型'
      }[this.selectedModel] || '未知模型';
    },
    fraudCount() {
      return this.predictions.filter(item => item.prediction === 'fraud').length;
    },
    actualFraudCount() {
      return this.predictions.filter(item => item.true_value === 1).length;
    }
  },
  methods: {
    async loadTestData() {
      this.error = null;
      this.loadingData = true;
      this.predictions = [];
      try {
        const response = await fetch('http://localhost:5000/api/load_test_data');
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || '加载测试集失败');
        }

        this.testSamples = result.records || [];
      } catch (err) {
        console.error('加载测试集失败:', err);
        this.error = err.message || '加载测试集时发生错误，请检查后端服务';
      } finally {
        this.loadingData = false;
      }
    },
    async predictModel() {
      if (!this.testSamples.length) {
        this.error = '请先加载测试集';
        return;
      }

      this.error = null;
      this.loadingPrediction = true;
      this.predictions = [];
      try {
        const response = await fetch('http://localhost:5000/api/predict_model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: this.selectedModel, limit: 100 })
        });
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || '模型预测失败');
        }

        this.predictions = result.results || [];
      } catch (err) {
        console.error('预测失败:', err);
        this.error = err.message || '预测时发生错误，请检查后端服务';
      } finally {
        this.loadingPrediction = false;
      }
    }
  }
};
</script>

<style scoped>
/* 整体页面容器：继承父级渐变背景，不设独立背景 */
.pfraud-page {
  padding: 24px 32px;
  max-width: 1400px;
  margin: 0 auto;
  background: linear-gradient(135deg, #e0f7fa 0%, #bbdefb 100%);
}

/* 页面标题区域 */
.page-title {
  margin-bottom: 32px;
}

.page-title h1 {
  color: #00796b;
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 8px;
  letter-spacing: -0.3px;
}

.page-title p {
  color: #37474f;
  font-size: 15px;
  opacity: 0.85;
  margin: 0;
}

/* 操作面板：毛玻璃风格面板 */
.actions-panel {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: flex-end;
  margin-bottom: 32px;
  background: rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(8px);
  border-radius: 20px;
  padding: 20px 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

/* 按钮通用样式 */
.primary-button,
.secondary-button {
  border: none;
  border-radius: 40px;
  padding: 10px 24px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.primary-button {
  background: linear-gradient(135deg, #2c7a6e, #1e5a5a);
  color: white;
}

.primary-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 18px rgba(0, 120, 100, 0.2);
  background: linear-gradient(135deg, #1e6b5e, #154e4e);
}

.secondary-button {
  background: linear-gradient(135deg, #1976d2, #0f5b9e);
  color: white;
  /* 关键：将开始预测按钮推到最右侧 */
  margin-left: auto;
}

.secondary-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 18px rgba(25, 118, 210, 0.2);
  background: linear-gradient(135deg, #1565c0, #0d4b82);
}

.primary-button:disabled,
.secondary-button:disabled {
  background: #b0bec5;
  transform: none;
  cursor: not-allowed;
  box-shadow: none;
}

/* 下拉选择组 — 加宽处理 */
.select-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;           /* 允许弹性增长 */
  max-width: 300px;  /* 限制最大宽度，避免过宽 */
}

.select-group label {
  font-size: 13px;
  font-weight: 500;
  color: #1e3a3a;
  letter-spacing: 0.3px;
}

.select-group select {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #cfdfe5;
  border-radius: 32px;
  padding: 10px 18px;
  font-size: 14px;
  color: #1e2a3a;
  cursor: pointer;
  transition: 0.2s;
  outline: none;
  width: 100%;       /* 使选择框填满父级宽度 */
}

.select-group select:focus {
  border-color: #2c7a6e;
  box-shadow: 0 0 0 2px rgba(44, 122, 110, 0.2);
}

/* 错误提示框 */
.error-box {
  background: rgba(255, 235, 238, 0.9);
  backdrop-filter: blur(4px);
  border-left: 5px solid #d32f2f;
  color: #b71c1c;
  padding: 14px 20px;
  border-radius: 16px;
  margin-bottom: 24px;
  font-weight: 500;
}

/* 卡片样式 — 毛玻璃效果，与参考保持一致 */
.card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(10px);
  border-radius: 28px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08);
  padding: 28px 32px;
  margin-bottom: 32px;
  transition: all 0.2s;
  border: 1px solid rgba(255, 255, 255, 0.4);
}

/* 卡片头部 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.card-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #1e3a3a;
  letter-spacing: -0.2px;
}

.card-header span {
  color: #3a5e5e;
  font-size: 14px;
  background: rgba(0, 0, 0, 0.04);
  padding: 4px 12px;
  border-radius: 40px;
}

/* 预测结果统计栏 */
.result-summary {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
  background: rgba(44, 122, 110, 0.1);
  padding: 12px 18px;
  border-radius: 20px;
  font-weight: 500;
  color: #1c4e46;
}

/* 表格容器及表格样式 */
.table-wrapper {
  overflow-x: auto;
  border-radius: 20px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 680px;
}

th,
td {
  padding: 14px 16px;
  text-align: left;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

th {
  background: rgba(44, 122, 110, 0.12);
  font-weight: 600;
  color: #1e3a3a;
  font-size: 14px;
}

td {
  color: #2c3e4e;
  font-size: 14px;
}

/* 结果表格悬停效果 */
.result-table tbody tr:hover {
  background: rgba(44, 122, 110, 0.06);
  transition: 0.1s;
}

/* 辅助文字 */
.note {
  margin-top: 20px;
  color: #547c7c;
  font-size: 13px;
  font-style: italic;
  padding-left: 6px;
}

/* 响应式调整 */
@media screen and (max-width: 900px) {
  .pfraud-page {
    padding: 20px;
  }
  .actions-panel {
    flex-direction: column;
    align-items: stretch;
  }
  /* 窄屏下移除按钮的自动右边距，恢复自然排列 */
  .secondary-button {
    margin-left: 0;
  }
  .select-group {
    max-width: none;  /* 窄屏下允许占满宽度 */
  }
  .card {
    padding: 20px;
  }
  .card-header {
    flex-direction: column;
    gap: 12px;
  }
  .result-summary {
    flex-direction: column;
    gap: 8px;
  }
}
</style>
