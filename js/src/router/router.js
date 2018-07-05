import Vue from 'vue'
import Router from 'vue-router'
import DetectionPage from '../components/page/detection_page/page.vue'
import PredictionPage from '../components/page/prediction_page/page.vue'
import DatasetsPage from '../components/page/datasets_page/page.vue'

Vue.use(Router)

const router = new Router({
  routes: [
    { path: '/', name: 'Training', component: DetectionPage },
    { path: '/prediction', name: 'Prediction', component: PredictionPage },
    { path: '/datasets', name: 'Datasets', component: DatasetsPage }
  ]
})

export default router
