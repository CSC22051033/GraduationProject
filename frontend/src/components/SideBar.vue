<template>
  <aside class="side-bar">
    <div class="logo">Dataset</div>

    <nav class="menu">
      <router-link to="/dataset/info" class="menu-item">数据集特征名称及描述</router-link>
      <router-link to="/dataset/exception" class="menu-item">数据集异常值处理</router-link>
      <router-link to="/dataset/balance" class="menu-item">数据集平衡</router-link>
      <router-link to="/dataset/feature"   class="menu-item">数据集特征筛选</router-link>
    </nav>
  </aside>
</template>

<script>
export default { name: 'SideBar' }
</script>

<style scoped>
.side-bar{
  width: 220px;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  background:#00796b;
  display: flex;
  flex-direction: column;
  padding: 24px 0;
  overflow-y: auto;
}
.logo{
  color:#fff;
  font-size: 20px;
  font-weight: 600;
  text-align: center;
  margin-bottom: 40px;
}
.menu{
  flex:1;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 16px;
}
.menu-item{
  color: rgba(255,255,255,.85);
  text-decoration: none;
  padding: 10px 12px;
  border-radius: 6px;
  transition: background .3s;
}
.menu-item:hover,
.menu-item.router-link-active{
  background: rgba(255,255,255,.15);
}
</style>