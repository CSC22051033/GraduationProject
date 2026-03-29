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

const routes = [
  {
    path: '/',
    name: 'HomePage',
    component: HomePage
  },
  {
    path: '/dataset',
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
    ]
  },
  {
    path: '/pFraud',
    component: () => import('@/views/Result.vue'),
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router