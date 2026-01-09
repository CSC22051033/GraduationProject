<template>
  <div class="card main-title">
    <h1>数据集获取</h1>
    <p>
      鉴于国内多数保险公司的用户信息均处于非公开状态，获取难度较大，而国外车险索赔数据较多。因此，本研究选用国外某保险公司的“carclaims”车险索赔数据集。
      该数据集包含 15420 条数据，每条数据共有 33 个属性，包含 32 个特征和 1 个标签。特征涵盖事故发生时间、申请索赔时间及投保人性别等相关信息。标签 FraudFound 中，
      0 代指正常索赔，1 表示欺诈索赔。
    </p>

  </div>
</template>

<script>
export default {
  name: 'DatasetIntro'
}
</script>

<style scoped>
/* 复用父组件的 card 样式即可，无需重复写 */
</style>