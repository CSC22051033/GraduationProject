<template>
  <div class="card table-card">
    <h2 class="table-title">数据集特征名称及描述</h2>

    <!-- 三栏容器 -->
    <div class="three-col">
      <div class="col" v-for="(chunk, cIdx) in chunks" :key="cIdx">
        <table class="feature-table">
          <thead>
            <tr>
              <th style="width:62px">序号</th>
              <th style="width:160px">特征名称</th>
              <th>特征描述</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, idx) in chunk" :key="idx">
              <td>{{ item.no }}</td>
              <td><span class="tag">{{ item.code }}</span></td>
              <td>{{ item.desc }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetInfo'  
}
</script>

<script setup>
// 原始 33 条数据
const allFeatures = [
  { no: 1,  code: 'Month',                  desc: '事故发生的月份' },
  { no: 2,  code: 'WeekOfMonth',            desc: '事故发生为第几周' },
  { no: 3,  code: 'DayOfWeek',              desc: '事故发生为星期几' },
  { no: 4,  code: 'Make',                   desc: '汽车制造商' },
  { no: 5,  code: 'AccidentArea',           desc: '事故发生的地理位置' },
  { no: 6,  code: 'DayOfWeekClaimed',       desc: '提出索赔为星期几' },
  { no: 7,  code: 'MonthClaimed',           desc: '提出索赔的月份' },
  { no: 8,  code: 'WeekOfMonthClaimed',     desc: '提出索赔为第几周' },
  { no: 9,  code: 'Sex',                    desc: '投保人的性别' },
  { no: 10, code: 'MaritalStatus',          desc: '投保人的婚姻状况' },
  { no: 11, code: 'Age',                    desc: '投保人的年龄' },
  { no: 12, code: 'Fault',                  desc: '肇事方' },
  { no: 13, code: 'Policy Type',            desc: '车辆类别和保单类型' },
  { no: 14, code: 'VehicleCategory',        desc: '车辆类型' },
  { no: 15, code: 'VehiclePrice',           desc: '车辆价格' },
  { no: 16, code: 'PolicyNumber',           desc: '保单编号' },
  { no: 17, code: 'RepNumber',              desc: '处理索赔的代理人编号' },
  { no: 18, code: 'Deductible',             desc: '免赔额' },
  { no: 19, code: 'DriverRating',           desc: '驾驶员评级' },
  { no: 20, code: 'Days_Policy_Accident',   desc: '事故发生时保单剩余天数' },
  { no: 21, code: 'Days_Policy_Claim',      desc: '提交索赔时保单剩余天数' },
  { no: 22, code: 'PastNumderOfClaims',     desc: '投保人的历史索赔次数' },
  { no: 23, code: 'AgeOfVehicle',           desc: '车辆使用寿命' },
  { no: 24, code: 'AgeOfPolicyHolder',      desc: '投保人年龄分组' },
  { no: 25, code: 'PoliceReportFiled',      desc: '是否向警方填写事故报告' },
  { no: 26, code: 'WitnessPresent',         desc: '是否有目击证人' },
  { no: 27, code: 'AgentType',              desc: '代理人类型' },
  { no: 28, code: 'NumberOfSuppliments',    desc: '附件或补充信息的数量' },
  { no: 29, code: 'AddressChange_Claim',    desc: '投保人在报告事故后搬家的时间' },
  { no: 30, code: 'NumberOfCars',           desc: '事故中涉及的车辆数量' },
  { no: 31, code: 'Year',                   desc: '事故发生的年份' },
  { no: 32, code: 'BasePolicy',             desc: '保单类型' },
  { no: 33, code: 'FraudFound',             desc: '是否发生欺诈（0=正常，1=欺诈）' }
]

// 计算属性：三栏均分
const chunkSize = Math.ceil(allFeatures.length / 3)
const chunks = [
  allFeatures.slice(0, chunkSize),
  allFeatures.slice(chunkSize, chunkSize * 2),
  allFeatures.slice(chunkSize * 2)
]
</script>

<style scoped>
.table-title {
  margin: 0 0 20px;
  color: #00796b;
  text-align: center;
}

.three-col {
  display: flex;
  gap: 20px;
}

.col {
  flex: 1;
  min-width: 0;
}

.feature-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  line-height: 1.5;
}

.feature-table thead {
  background: #00796b;
  color: #fff;
}

.feature-table th,
.feature-table td {
  padding: 8px 10px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  white-space: nowrap;
}

.feature-table tbody tr:nth-child(even) {
  background: rgba(0, 121, 107, 0.04);
}

.feature-table tbody tr:hover {
  background: rgba(0, 121, 107, 0.08);
}

.tag {
  display: inline-block;
  background: rgba(0, 121, 107, 0.12);
  color: #00796b;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-family: 'Consolas', 'Monaco', monospace;
}

/* 小屏自动堆叠 */
@media (max-width: 1024px) {
  .three-col {
    flex-direction: column;
  }
}
</style>

