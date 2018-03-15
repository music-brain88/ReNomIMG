import Vue from 'vue'
import store from './store/store'
import router from './router/router'
import App from './app'

new Vue({
    el: '#app',
    router: router,
    store: store,
    render: h => h(App)
})
