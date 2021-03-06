import Model from './classes/model'
import axios from 'axios'
import { STATE } from './../const.js'
import { Dataset } from './classes/dataset'

function error_handler_creator (context, callback = undefined) {
  return function (error) {
    const status = error.response.status
    if ([200, 400, 403, 404, 405, 500, 503].includes(status)) {
      if (error.response.data && error.response.data.error) {
        const message = error.response.data.error.message
        context.commit('showAlert', '【' + status + ' Error】: ' + message)
      }
    }
    if (callback) {
      callback()
    }
  }
}

export default {
  /** ***
   *
   */
  async init (context, payload) {
    context.commit('showLoadingMask', true)
    // context.commit('resetState')
    context.commit('flushFilter')
    context.dispatch('loadDatasetsOfCurrentTask')
    await context.dispatch('loadModelsOfCurrentTask', 'all')
    await context.dispatch('loadModelsOfCurrentTask', 'deployed')
    context.commit('showLoadingMask', false)
    context.dispatch('startAllPolling')
  },

  /** ***
   * Get models list (No details)
   */
  async loadModelsOfCurrentTask (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const m_state = payload
    const url = '/renom_img/v2/api/' + task_name + '/models?state=' + m_state
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        const model_list = response.data.models
        for (const m of model_list) {
          const algorithm_id = m.algorithm_id
          const task_id = m.task_id
          const state = m.state
          const id = m.id
          const hyper_params = m.hyper_parameters
          const dataset_id = m.dataset_id
          const model = new Model(algorithm_id, task_id, hyper_params, dataset_id)

          model.id = id
          model.state = state
          model.total_epoch = m.total_epoch
          model.nth_epoch = m.nth_epoch
          model.total_batch = m.total_batch
          model.nth_batch = m.nth_batch
          model.train_loss_list = m.train_loss_list
          model.valid_loss_list = m.valid_loss_list
          model.best_epoch_valid_result = m.best_epoch_valid_result
          model.last_batch_loss = m.last_batch_loss
          model.last_prediction_result = m.last_prediction_result

          if (m_state !== 'deployed') {
            context.commit('addModel', model)
          } else {
            context.commit('updateModel', model)
            const deployed_model = context.getters.getModelById(model.id)
            context.commit('setDeployedModel', deployed_model)
          }
          // context.dispatch('forceUpdatePage', id)
        }
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async loadModelsOfCurrentTaskDetail (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        const m = response.data.model

        const algorithm_id = m.algorithm_id
        const task_id = m.task_id
        const state = m.state
        const id = m.id
        const hyper_params = m.hyper_parameters
        const dataset_id = m.dataset_id
        const model = new Model(algorithm_id, task_id, hyper_params, dataset_id)

        model.id = id
        model.state = state
        model.total_epoch = m.total_epoch
        model.nth_epoch = m.nth_epoch
        model.total_batch = m.total_batch
        model.nth_batch = m.nth_batch
        model.train_loss_list = m.train_loss_list
        model.valid_loss_list = m.valid_loss_list
        model.best_epoch_valid_result = m.best_epoch_valid_result
        model.last_batch_loss = m.last_batch_loss
        model.last_prediction_result = m.last_prediction_result
        context.commit('updateModel', model)

        // TODO muraishi : 呼び出し元でやる
        // context.commit('setSelectedModel', model)
        // context.dispatch('forceUpdatePage', model.id)
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async loadDatasetsOfCurrentTask (context) {
    const task_name = context.getters.getCurrentTaskName
    const url = '/renom_img/v2/api/' + task_name + '/datasets'
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        for (const ds of response.data.datasets) {
          const id = ds.id
          const class_map = ds.class_map
          const valid_data = ds.valid_data
          const task = ds.task_id
          const name = ds.name
          const ratio = ds.ratio
          const description = ds.description
          const test_dataset_id = ds.test_dataset_id
          const class_info = ds.class_info
          const loaded_dataset = new Dataset(task, name, ratio, description, test_dataset_id)
          loaded_dataset.id = id
          loaded_dataset.class_map = class_map
          loaded_dataset.valid_data = valid_data
          loaded_dataset.class_info = class_info
          context.commit('addDataset', loaded_dataset)
        }
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async loadDatasetsOfCurrentTaskDetail (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const dataset_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/datasets/' + dataset_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        const ds = response.data.dataset

        const id = ds.id
        const class_map = ds.class_map
        const valid_data = ds.valid_data
        const task = ds.task_id
        const name = ds.name
        const ratio = ds.ratio
        const description = ds.description
        const test_dataset_id = ds.test_dataset_id
        const class_info = ds.class_info
        const loaded_dataset = new Dataset(task, name, ratio, description, test_dataset_id)
        loaded_dataset.id = id
        loaded_dataset.class_map = class_map
        loaded_dataset.valid_data = valid_data
        loaded_dataset.class_info = class_info
        context.commit('updateDataset', loaded_dataset)
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async createModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const url = '/renom_img/v2/api/' + task_name + '/models'
    const hyper_params = payload.hyper_params
    const algorithm_id = payload.algorithm_id
    const dataset_id = payload.dataset_id
    const task_id = payload.task_id

    return axios.post(url, {
      hyper_parameters: hyper_params,
      dataset_id: dataset_id,
      task_id: task_id,
      algorithm_id: algorithm_id,
    })
      .then(function (response) {
        if (response.status === 204) return
        const id = response.data.model.id
        context.dispatch('runTrainThread', {
          model_id: id,
          hyper_params: hyper_params,
          algorithm_id: algorithm_id,
          dataset_id: dataset_id,
          task_id: task_id
        })
      }, error_handler_creator(context))
  },
  /** ***
   *
   */
  async removeModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model_id
    return axios.delete(url)
      .then(function (response) {
        // TODO: if (response.status === 204) return
        context.commit('rmModel', model_id)
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async runTrainThread (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload.model_id
    const hyper_params = payload.hyper_params
    const algorithm_id = payload.algorithm_id
    const dataset_id = payload.dataset_id
    const task_id = payload.task_id
    const url = '/renom_img/v2/api/' + task_name + '/train'

    return axios.post(url, {
      model_id: model_id
    })
      .then(function (response) {
        const model = new Model(algorithm_id, task_id, hyper_params, dataset_id)
        model.id = model_id
        model.state = STATE.CREATED
        context.commit('addModel', model)
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async pollingTrain (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/train?model_id=' + model_id
    const request_source = { 'train': model_id }
    const current_requests = context.state.polling_request_jobs.train

    // If polling for the model is already performed, do nothing.
    if (current_requests.indexOf(model_id) >= 0) {
      return
    }

    // Register request.
    context.commit('addPollingJob', request_source)
    return axios.get(url).then(function (response) {
      // This 'response' can be empty.

      // Check and run other model's polling.
      context.dispatch('startAllPolling', payload)

      // Remove finished request.
      context.commit('rmPollingJob', request_source)

      // Need to confirm the model is not removed.
      const model = context.getters.getModelById(model_id)
      if (model) {
        const r = response.data
        const state = r.state
        const load_best = response.data.best_result_changed

        // Update model.
        model.state = r.state
        model.running_state = r.running_state
        model.total_epoch = r.total_epoch
        model.nth_epoch = r.nth_epoch
        model.total_batch = r.total_batch
        model.nth_batch = r.nth_batch
        model.last_batch_loss = r.last_batch_loss
        model.train_loss_list = (r.train_loss_list) ? r.train_loss_list : []
        model.valid_loss_list = (r.valid_loss_list) ? r.valid_loss_list : []
        model.best_epoch_valid_result = r.best_epoch_valid_result

        if (state === STATE.STOPPED) {

        } else {
          context.dispatch('pollingTrain', model_id)
        }
        if (load_best) {
          context.commit('forceUpdateModelList')
          context.commit('forceUpdatePredictionPage')
          // context.dispatch('updateBestValidResult', model_id) // 使用禁止
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },

  async runPredictionThread (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/prediction'

    const model = context.getters.getModelById(model_id)
    model.state = STATE.PRED_CREATED // TODO: Remove this line.
    model.total_prediction_batch = 0
    model.nth_prediction_batch = 0
    return axios.post(url, {
      model_id: model_id
    })
      .then(function (response) {
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
  },

  async pollingPrediction (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/prediction?model_id=' + model_id
    const request_source = { 'prediction': model_id }
    const current_requests = context.state.polling_request_jobs.prediction
    // If polling for the model is already performed, do nothing.
    if (current_requests.indexOf(model_id) >= 0) {
      return
    }

    // Register request.
    context.commit('addPollingJob', request_source)
    return axios.get(url).then(function (response) {
      // This 'response' can be empty.
      // Check and run other model's polling.
      context.dispatch('startAllPolling', payload)

      // Remove finished request.
      context.commit('rmPollingJob', request_source)

      // Need to confirm the model is not removed.
      const model = context.getters.getModelById(model_id)
      if (model) {
        const r = response.data
        const state = r.state
        const need_pull = response.data.need_pull
        // Update model.
        model.state = r.state
        model.running_state = r.running_state
        model.total_prediction_batch = r.total_batch
        model.nth_prediction_batch = r.nth_batch

        if (state === STATE.STOPPED) {

        } else {
          context.dispatch('pollingPrediction', model_id)
        }
        if (need_pull) {
          context.dispatch('updatePredictionResult', model_id)
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },

  /* 不要な処理のため使用しない
  async updateBestValidResult (context, payload) {
    const model_id = payload
    const old_model = context.getters.getModelById(model_id)

    if (old_model) {
      await context.dispatch('loadModelsOfCurrentTaskDetail', model_id)
      const new_model = context.getters.getModelById(model_id)

      if (context.state.selected_model &&
          context.state.selected_model.id === model_id) {
        context.commit('setSelectedModel', new_model)
      }
      context.commit('forceUpdateModelList')
      context.commit('forceUpdatePredictionPage')
    }
  },
  */

  async updatePredictionResult (context, payload) {
    const model_id = payload
    const old_deployed_model = context.getters.getModelById(model_id)

    if (old_deployed_model) {
      await context.dispatch('loadModelsOfCurrentTaskDetail', model_id)
      const new_deployed_model = context.getters.getModelById(model_id)

      context.commit('setDeployedModel', new_deployed_model)
      context.commit('forceUpdateModelList')
      context.commit('forceUpdatePredictionPage')
    }
  },

  async forceUpdatePage (context, payload) {
    const model_id = payload
    const model = context.getters.getModelById(model_id)
    if (model) {
      context.commit('forceUpdateModelList')
      context.commit('forceUpdatePredictionPage')
    }
  },

  async updateSelectedModel (context, payload) {
    const model = payload
    context.dispatch('loadDatasetsOfCurrentTaskDetail', model.dataset_id)
    await context.dispatch('loadModelsOfCurrentTaskDetail', model.id)
    // set selected_model form updated state.models
    const selected_model = context.getters.getModelById(model.id)
    context.commit('setSelectedModel', selected_model)
  },

  /** ***
   *
   */
  async stopModelTrain (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/train'

    return axios.delete(url, {
      data: { model_id: model_id }
    }).then(function (response) {

    }, error_handler_creator(context))
  },

  /** ***
   * disern the kind of job (Train or Predict)
   */
  async startAllPolling (context, payload) {
    const model_list = context.state.models.filter(m => m.id >= 0)
    const current_train_requests = context.state.polling_request_jobs.train
    const current_prediction_requests = context.state.polling_request_jobs.prediction

    for (const m of model_list) {
      if (
        m.state === STATE.CREATED ||
        m.state === STATE.RESERVED ||
        m.state === STATE.STARTED
      ) {
        let need_run_request = true
        for (const model_id in current_train_requests) {
          if (model_id === m.id) {
            need_run_request = false
          }
        }

        if (need_run_request) {
          context.dispatch('pollingTrain', m.id)
        }
      } else if (
        m.state === STATE.PRED_CREATED ||
        m.state === STATE.PRED_RESERVED ||
        m.state === STATE.PRED_STARTED
      ) {
        let need_run_request = true
        for (const model_id in current_prediction_requests) {
          if (model_id === m.id) {
            need_run_request = false
          }
        }
        if (need_run_request) {
          context.dispatch('pollingPrediction', m.id)
        }
      }
    }
  },

  /** ***
   * PUT the tempDataset : not using in current version v2.2
   */
  async createDataset (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const dataset_id = payload.dataset_id
    const url = '/renom_img/v2/api/' + task_name + '/datasets/' + dataset_id

    return axios.put(url).then(function (response) {
      if (response.status === 204) return
      context.dispatch('loadDatasetsOfCurrentTaskDetail', dataset_id)
    }, error_handler_creator(context))
  },

  /** ***
   * POST and create the tempDataset
   */
  async confirmDataset (context, payload) {
    // TODO: const url = '/api/renom_img/v2/dataset/confirm'
    const task_name = context.getters.getCurrentTaskName
    const url = '/renom_img/v2/api/' + task_name + '/datasets'

    const hash = payload.hash
    const name = payload.name
    const test_dataset_id = payload.test_dataset_id
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = payload.description

    return axios.post(url, {
      name: encodeURIComponent(name),
      hash: hash,
      ratio: ratio,
      task_id: task_id,
      description: encodeURIComponent(description),
      test_dataset_id: test_dataset_id
    }).then(function (response) {
      if (response.status === 204) return
      const class_map = response.data.dataset.class_map
      const valid_data = response.data.dataset.valid_data
      const class_info = response.data.dataset.class_info
      const id = response.data.dataset.id

      // The dataset id will be available when the dataset registered to DB.
      // So tentatively, insert -1.
      const dataset = new Dataset(task_id, name, ratio, description, test_dataset_id)
      dataset.class_map = class_map
      dataset.valid_data = valid_data
      dataset.class_info = class_info
      dataset.id = id

      // TODO: console.dir(dataset)
      context.commit('setConfirmingDataset', dataset)
      context.commit('setConfirmingFlag', false)
    }, error_handler_creator(context, () => {
      context.commit('setConfirmingDataset', null)
      context.commit('setConfirmingFlag', false)
    }))
  },

  /** ***
   *
   */
  async deleteDataset (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const dataset_id = payload
    const url = '/renom_img/v2/api/' + task_name + '/datasets/' + dataset_id

    return axios.delete(url).then(function (response) {
      context.commit('setConfirmingDataset', null)
      context.commit('setConfirmingFlag', false)
    }, error_handler_creator(context, () => {
      context.commit('setConfirmingDataset', null)
      context.commit('setConfirmingFlag', false)
    }))
  },

  async loadSegmentationTargetArray (context, payload) {
    const dataset_id = payload.dataset_id
    const name = payload.name
    const width = payload.size.width
    const height = payload.size.height
    const query = 'filename=' + name + '&width=' + width + '&height=' + height
    const url = '/renom_img/v2/api/segmentation/datasets/' + dataset_id + '/mask?' + query

    const callback = payload.callback
    return axios.get(url).then(response => {
      callback(response)
    }, error_handler_creator(context))
  },
  async deployModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model.id
    return axios.put(url, {
      deploy: true
    }).then(function (response) {
      context.commit('setDeployedModel', model)
    }, error_handler_creator(context))
  },
  async unDeployModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model.id
    return axios.put(url, {
      deploy: false
    }).then(function (response) {
      context.commit('unDeployModel')
    }, error_handler_creator(context))
  },

  async downloadPredictionResult (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    const format = 'csv'// 現状はcsvのみですが、今後増える想定です。
    const url = '/renom_img/v2/api/' + task_name + '/prediction/result?model_id=' + model.id + '&format=' + format
    window.open(url, '__blank')
  },
}
