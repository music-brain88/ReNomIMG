import Vue from 'vue'
import Router from 'vue-router'

// import DebugPage from './../components/page/debug_page/page.vue'

// ↓DatasetPageはOrganismsが1つしかないため、直接指定しています。
import TrainPage from './../components/Pages/ReNomIMG/rnc-train/rnc-train.vue'
import DatasetPage from './../components/Organisms/rnc-dataset-panel-dataset/rnc-dataset-panel-dataset.vue'
import PredictionPage from './../components/Pages/ReNomIMG/rnc-predict/rnc-predict.vue'

Vue.use(Router)
const router = new Router({
  routes: [
    // { path: '/debug', component: DebugPage },
    { path: '/', name: 'Training', component: TrainPage },
    { path: '/dataset', name: 'Dataset', component: DatasetPage },
    { path: '/predict', name: 'Prediction', component: PredictionPage }
  ]
})

export default router
