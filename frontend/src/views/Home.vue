<template>
  <section class="home">
    <div class="card main-title">
      <h1>基于深度学习的车险欺诈识别系统设计与实现</h1>
    </div>

    <div class="card sub-title">
      <p>制作人：陈帅驰，指导老师：魏丹</p>
    </div>
  </section>
</template>

<script>
export default { name: 'HomePage' }
</script>

<style scoped>
/* 全屏居中容器 */
.home {
  min-height: 100vh;               /* 占满可视高度 */
  display: flex;
  flex-direction: column;
  justify-content: center;         /* 垂直居中 */
  align-items: center;             /* 水平居中 */
  background: linear-gradient(135deg, #e0f7fa 0%, #bbdefb 100%);
  padding: 40px 20px;
  box-sizing: border-box;
}

/* 卡片通用样式 */
.card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(10px);     /* 毛玻璃效果 */
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  padding: 32px 48px;
  margin: 12px 0;
  text-align: center;
  max-width: 800px;
  width: 100%;
}

/* 标题样式 */
.main-title h1 {
  color: #00796b;                  /* 深一点的翡翠绿 */
  font-size: 2.2rem;
  font-weight: 600;
  line-height: 1.4;
  margin: 0;
}

/* 副标题样式 */
.sub-title p {
  color: #004d40;
  font-size: 1.3rem;
  margin: 0;
}

/* 响应式：窄屏自动缩小 */
@media screen and (max-width: 600px) {
  .main-title h1 { font-size: 1.6rem; }
  .sub-title p  { font-size: 1.1rem; }
  .card         { padding: 24px 16px; }
}
</style>