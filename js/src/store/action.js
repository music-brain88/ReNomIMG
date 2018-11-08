import Model from './classes/model'
import axios from 'axios'
import { STATE } from '@/const.js'

export default {
  /*****
   *
   */
  async init (context, payload) {
    context.dispatch('loadModelsOfCurrentTask')
  },

  /*****
   *
   */
  async loadModelsOfCurrentTask (context, payload) {
    let task = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/model/load/task/' + task
    return axios.get(url)
      .then(function (response) {
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
        } else {
          let model_list = response.data.model_list
          for (let m of model_list) {
            let algorithm_id = m.algorithm_id
            let task_id = m.task_id
            let state = m.state
            let id = m.id
            let hyper_params = m.hyper_parameters
            let parents = []
            let dataset_id = m.dataset_id
            let model = new Model(algorithm_id, task_id, hyper_params, dataset_id, parents)
            model.id = id
            context.commit('addModel', model)
          }
        }
      })
  },

  /*****
   *
   */
  async loadDataset (context, payload) {
    const url = '/api/renom_img/v1/projects/' + payload.project_id
    return axios.get(url)
      .then(function (response) {
      })
  },

  /*****
   *
   */
  async createModel (context, payload) {
    const url = '/api/renom_img/v2/model/create'
    const hyper_params = payload.hyper_params
    const algorithm_id = payload.algorithm_id
    const dataset_id = payload.dataset_id
    const parents = payload.parents
    const task_id = payload.task_id
    const param = new FormData()
    const model = new Model(algorithm_id, task_id, hyper_params, dataset_id, parents)
    context.commit('addModel', model)

    // Append params.
    param.append('hyper_params', JSON.stringify(hyper_params))
    param.append('parents', JSON.stringify(parents))
    param.append('dataset_id', dataset_id)
    param.append('task_id', task_id)
    param.append('algorithm_id', algorithm_id)

    return axios.post(url, param)
      .then(function (response) {
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
        } else {
          let id = response.data.id
          model.id = id
          model.state = STATE.CREATED
        }
      })
  },

  /*****
   *
   */
  async createDataset (context, payload) {
    const param = new FormData()
    param.append('dataset_def_id', payload.dataset_def_id)
    return axios.post(url, param)
  },
}
