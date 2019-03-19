import Vue from 'vue'
import Vuex from 'vuex'
import state from './state'
import getter from './getter'
import mutation from './mutation'
import action from './action'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: state,
  getters: getter,
  mutations: mutation,
  actions: action
})

export default store
