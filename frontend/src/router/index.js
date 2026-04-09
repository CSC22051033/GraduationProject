// router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '@/views/Home.vue'

import DatasetIntro from '@/views/dataset/Intro.vue'
import DatasetInfo from '@/views/dataset/Info.vue'
import DatasetException from '@/views/dataset/Exception.vue'
import DatasetBalance from '@/views/dataset/Balance.vue'
import DatasetFeature from '@/views/dataset/Feature.vue'

import VisualCNN from '@/views/visual/CNN.vue'
import VisualRNN from '@/views/visual/RNN.vue'
import VisualMIX from '@/views/visual/MIX.vue'
import PFraud from '@/views/PFraud.vue'

const routes = [
  {
    path: '/',
    name: 'HomePage',
    component: HomePage
  },
  {
    path: '/dataset',
    // 懒加载已足够，无需再声明 DatasetPage 变量
    component: () => import('@/views/Dataset.vue'),
    redirect: '/dataset/intro',
    children: [
      { path: 'intro', component: DatasetIntro },
      { path: 'info',  component: DatasetInfo  },
      { path: 'exception',  component: DatasetException  },
      { path: 'balance',  component: DatasetBalance  },
      { path: 'feature',  component: DatasetFeature }
    ]
  },
  {
    path: '/visual',
    component: () => import('@/views/Visual.vue'),
    children: [
      { path: 'cnn', component: VisualCNN },
      { path: 'rnn', component: VisualRNN },
      { path: 'mix', component: VisualMIX },
    ]
  },
  {
    path: '/pFraud',
    name: 'PFraud',
    component: PFraud
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router