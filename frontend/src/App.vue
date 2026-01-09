<template>
  <div id="app">
    <TopBar />          <!-- 1. 放在最上面 -->
    <main class="main"> <!-- 2. 给页面内容留顶部间距 -->
      <router-view />
    </main>
  </div>
</template>

<script>
import TopBar from '@/components/TopBar.vue'

export default {
  components: { TopBar }
}
</script>

<style>
/* 让内容不被顶部栏遮住 */
.main { margin-top: 56px; }
</style>