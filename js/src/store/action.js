import Model from './classes/model'
import axios from 'axios'
import { STATE } from '@/const.js'

export default {
  /*****
   *
   */
  async init (context, payload) {
    await context.dispatch('loadModelsOfCurrentTask')
    context.dispatch('startAllPolling')
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
            model.total_epoch = m.total_epoch
            model.nth_epoch = m.nth_epoch
            model.total_batch = m.total_batch
            model.nth_batch = m.nth_batch
            model.train_loss_list = m.train_loss_list
            model.valid_loss_list = m.valid_loss_list
            model.best_epoch_valid_result = m.best_epoch_valid_result
            model.last_batch_loss = m.last_batch_loss

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
    model.state = STATE.CREATED

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
          model.state = STATE.RESERVED
        }
      })
  },

  /*****
   *
   */
  async runTrainThread (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/thread/run/' + model_id
    return axios.get(url)
      .then(function (response) {
        let model = context.getters.getModelById(model_id)
        model.state = STATE.CREATED // TODO: Remove this line.
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
        }
        context.dispatch('startAllPolling')
      })
  },

  /*****
   *
   */
  async pollingTrain (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/polling/train/model/' + model_id
    const request_source = {'train': model_id}
    const current_requests = context.state.polling_request_jobs.train

    // If polling for the model is already performed, do nothing.
    if (current_requests.indexOf(model_id) >= 0) {
      return
    }

    // Register request.
    context.commit('addPollingJob', request_source)
    return axios.get(url).then(function (response) {
      // Check and run other model's polling.
      context.dispatch('startAllPolling', payload)

      // Remove finished request.
      context.commit('rmPollingJob', request_source)
      let error_msg = response.data.error_msg
      if (error_msg) {
        context.commit('showAlert', {'show': true, 'msg': error_msg})
        return
      }

      const model = context.getters.getModelById(model_id)
      let r = response.data
      let state = r.state
      let load_best = response.data.best_result_changed

      // Update model.
      model.state = r.state
      model.running_state = r.running_state
      model.total_epoch = r.total_epoch
      model.nth_epoch = r.nth_epoch
      model.total_batch = r.total_batch
      model.nth_batch = r.nth_batch
      model.last_batch_loss = r.last_batch_loss

      if (state === STATE.STOPPED) {

      } else {
        context.dispatch('pollingTrain', model_id)
      }
      if (load_best) {
        context.dispatch('loadBestValidResult', model_id)
      }
    })
  },
  /*****
   *
   */
  async loadBestValidResult (context, payload) {

  },

  /*****
   *
   */
  async startWeightDownload (context, payload) {
    const algorithm_id = payload
  },

  /*****
   *
   */
  async pollingWeightDownload (context, payload) {

  },

  /*****
   *
   */
  async startAllPolling (context, payload) {
    const model_list = context.state.models
    const current_requests = context.state.polling_request_jobs.train
    for (let m of model_list) {
      if (m.state !== STATE.STOPPED) {
        let need_run_request = true
        for (let model_id in current_requests) {
          if (model_id === m.id) {
            need_run_request = false
          }
        }
        if (need_run_request) {
          context.dispatch('pollingTrain', m.id)
        }
      }
    }
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
