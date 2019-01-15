import Vue from 'vue'
import Router from 'vue-router'
import DebugPage from '../components/page/debug_page/page.vue'
import PredictionPage from '../components/page/prediction_page/page.vue'
import TrainPage from '../components/page/train_page/page.vue'
import DatasetPage from '../components/page/dataset_page/page.vue'

Vue.use(Router)
const router = new Router({
  routes: [
    { path: '/', component: TrainPage },
    { path: '/debug', component: DebugPage },
    { path: '/predict', component: PredictionPage },
    { path: '/dataset', component: DatasetPage },
  ]
})

export default router
