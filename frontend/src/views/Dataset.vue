<template>
  <div class="dataset-container">
    <SideBar />
    <main class="dataset-main">
      <section class="dataset-page">
        <router-view />   <!-- 先渲染 intro，点后渲染 info -->
      </section>
    </main>
  </div>
</template>

<script>
import SideBar from '@/components/SideBar.vue'
export default {
  name: 'DatasetLayout', 
  components: { SideBar }
}
</script>

<style scoped>
.dataset-container{
  display: flex;
  height: 100vh;
}
.dataset-main{
  flex: 1;
  margin-left: 220px;   /* 侧边栏宽度 */
  overflow-y: auto;
}
.dataset-page{
  min-height: 100vh;
  background: linear-gradient(135deg, #e0f7fa 0%, #bbdefb 100%);
  padding: 40px 20px;
  box-sizing: border-box;
}
h1{ color: #00796b; }

.card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(10px);     /* 毛玻璃效果 */
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  padding: 32px 48px;
  margin: 12px 0;
  max-width: 1300px;
  width: 100%;
}

</style>