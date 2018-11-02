import Vue from 'vue'
import Router from 'vue-router'
import DebugPage from '../components/page/debug_page/page.vue'

Vue.use(Router)
const router = new Router({
  routes: [
    { path: '/', name: 'DEBUG', component: DebugPage }
  ]
})

export default router
