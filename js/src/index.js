import Vue from 'vue'
import store from './store/store'
import router from './router/router'
import App from './app'

Vue.config.devtools = true
new Vue({
  el: '#app',
  router: router,
  store: store,
  render: h => h(App)
})
