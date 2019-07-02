import Vue from 'vue'
import Router from 'vue-router'

// import DebugPage from './../components/page/debug_page/page.vue'

// ↓DatasetPageはOrganismsが1つしかないため、直接指定しています。
import DatasetPage from './../components/Organisms/rnc-dataset-panel-dataset/rnc-dataset-panel-dataset.vue'
import PredictionPage from './../components/Pages/ReNomIMG/rnc-predict/rnc-predict.vue'
import TrainPage from './../components/Pages/ReNomIMG/rnc-train/rnc-train.vue'

Vue.use(Router)
const router = new Router({
  routes: [
    // { path: '/debug', component: DebugPage },
    { path: '/', component: TrainPage },
    { path: '/predict', component: PredictionPage },
    { path: '/dataset', component: DatasetPage }
  ]
})

export default router
