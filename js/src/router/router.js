import Vue from 'vue'
import Router from 'vue-router'
import DebugPage from '../components/page/debug_page/page.vue'
import TrainPage from '../components/page/train_page/page.vue'

Vue.use(Router)
const router = new Router({
  routes: [
    { path: '/', name: 'Train', component: TrainPage },
    { path: '/debug', name: 'DEBUG', component: DebugPage },
  ]
})

export default router
