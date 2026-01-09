<template>
  <header class="top-bar">
    <!-- 左侧 Logo / 标题 -->
    <div class="left">
      <router-link to="/" class="logo">基于深度学习的车险欺诈识别系统设计与实现</router-link>
    </div>

    <!-- 中间导航 -->
    <nav class="nav">
      <router-link to="/">首页</router-link>
      <router-link to="/dataset">数据集分析</router-link>
      <router-link to="/visual">训练历史可视化</router-link>
      <router-link to="/pFraud">欺诈概率输出</router-link>
    </nav>

  </header>
</template>

<script>
export default { name: 'TopBar' }
</script>

<style scoped>
/* 顶部栏整体 */
.top-bar {
  position: fixed;          /* 固定在视口顶部 */
  top: 0;
  left: 0;
  right: 0;
  height: 56px;
  background: #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,.08);
  display: flex;
  align-items: center;
  padding: 0 24px;
  z-index: 1000;
}

/* 三块布局 */
.left   { flex: 0 0 400px; }
.nav    { flex: 1; display: flex; gap: 24px; justify-content:end; }

/* 文字样式 */
.logo {
  font-size: 20px;
  font-weight: 600;
  color: #42b983;
  text-decoration: none;
}
.nav a {
  color: #2c3e50;
  text-decoration: none;
  font-size: 18px;
}
.nav a.router-link-active { color: #42b983; }

</style>