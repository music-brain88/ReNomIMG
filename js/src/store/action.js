import Model from './classes/model'
import axios from 'axios'
import C from '../const.js'

export default {
  /*****
   *
   */
  async init (context, payload) {
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
    const dataset_id = payload.dataset_id
    const parents = payload.parents
    const task = payload.task
    const param = new FormData()
    const model = new Model(undefined, task, hyper_params, dataset_id, parents)
    context.commit('addModel', model)

    // Append params.
    param.append('hyper_params', JSON.stringify(hyper_params))
    param.append('parents', JSON.stringify(parents))
    param.append('dataset_id', dataset_id)
    param.append('task', task)

    return axios.post(url, param)
      .then(function (response) {
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
        } else {
          let id = response.data.id
          model.id = id
          model.state = C.STATE.CREATED
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
