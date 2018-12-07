import Model from './classes/model'
import axios from 'axios'
import { STATE } from '@/const.js'
import { Dataset, TestDataset } from './classes/dataset'

function error_handler_creator (context, callback = undefined) {
  return function (error) {
    const status = error.response.status
    if (status === 500) {
      const message = error.response.data.error_msg
      context.commit('showAlert', message)
    }
    if (callback) {
      callback()
    }
  }
}

export default {
  /*****
   *
   */
  async init (context, payload) {
    context.commit('showLoadingMask', true)
    // context.commit('resetState')
    context.commit('flushFilter')
    context.dispatch('loadDatasetsOfCurrentTask')
    context.dispatch('loadTestDatasetsOfCurrentTask')
    await context.dispatch('loadModelsOfCurrentTask')
    await context.dispatch('loadDeployedModel')
    context.commit('showLoadingMask', false)
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
        if (response.status === 204) return
        let model_list = response.data.model_list
        for (let m of model_list) {
          let algorithm_id = m.algorithm_id
          let task_id = m.task_id
          let state = m.state
          let id = m.id
          let hyper_params = m.hyper_parameters
          let dataset_id = m.dataset_id
          let model = new Model(algorithm_id, task_id, hyper_params, dataset_id)

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

          context.commit('addModel', model)
          context.dispatch('loadBestValidResult', id)
        }
      }, error_handler_creator(context))
  },

  /*****
   *
   */
  async loadDatasetsOfCurrentTask (context) {
    const task_id = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/dataset/load/task/' + task_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        for (let ds of response.data.dataset_list) {
          const id = ds.id
          const class_map = ds.class_map
          const valid_data = ds.valid_data
          const task = ds.task_id
          const name = ds.name
          const ratio = ds.ratio
          const description = ds.description
          const test_dataset_id = ds.test_dataset_id
          const loaded_dataset = new Dataset(task, name, ratio, description, test_dataset_id)
          loaded_dataset.id = id
          loaded_dataset.class_map = class_map
          loaded_dataset.valid_data = valid_data
          context.commit('addDataset', loaded_dataset)
        }
      }, error_handler_creator(context))
  },

  /*****
   *
   */
  async loadTestDatasetsOfCurrentTask (context) {
    const task_id = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/test_dataset/load/task/' + task_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        for (let ds of response.data.test_dataset_list) {
          const id = ds.id
          const class_map = ds.class_map
          const test_data = ds.test_data
          const task = ds.task_id
          const name = ds.name
          const ratio = ds.ratio
          const description = ds.description
          const loaded_dataset = new TestDataset(task, name, ratio, description)
          loaded_dataset.id = id
          loaded_dataset.class_map = class_map
          loaded_dataset.test_data = test_data
          context.commit('addTestDataset', loaded_dataset)
        }
      }, error_handler_creator(context))
  },

  /*****
   *
   */
  async createModel (context, payload) {
    const url = '/api/renom_img/v2/model/create'
    const hyper_params = payload.hyper_params
    const algorithm_id = payload.algorithm_id
    const dataset_id = payload.dataset_id
    const task_id = payload.task_id
    const param = new FormData()
    const model = new Model(algorithm_id, task_id, hyper_params, dataset_id)
    context.commit('addModel', model)
    model.state = STATE.CREATED

    // Append params.
    param.append('hyper_params', JSON.stringify(hyper_params))
    param.append('dataset_id', dataset_id)
    param.append('task_id', task_id)
    param.append('algorithm_id', algorithm_id)

    return axios.post(url, param)
      .then(function (response) {
        if (response.status === 204) return
        let id = response.data.id
        model.id = id
        model.state = STATE.RESERVED
        context.dispatch('runTrainThread', id)
      }, error_handler_creator(context))
  },
  /*****
   *
   */
  async removeModel (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/remove/' + model_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        context.commit('rmModel', model_id)
      }, error_handler_creator(context))
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
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
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
      // This 'response' can be empty.

      // Check and run other model's polling.
      context.dispatch('startAllPolling', payload)

      // Remove finished request.
      context.commit('rmPollingJob', request_source)

      // Need to confirm the model is not removed.
      const model = context.getters.getModelById(model_id)
      if (model) {
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
        model.train_loss_list = (r.train_loss_list) ? r.train_loss_list : []
        model.valid_loss_list = (r.valid_loss_list) ? r.valid_loss_list : []

        if (state === STATE.STOPPED) {

        } else {
          context.dispatch('pollingTrain', model_id)
        }
        if (load_best) {
          context.dispatch('loadBestValidResult', model_id)
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },
  /*****
   *
   */
  async loadBestValidResult (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/load/best/result/' + model_id
    return axios.get(url).then(function (response) {
      const model = context.getters.getModelById(model_id)
      if (model) {
        const r = response.data
        const best_result = r.best_result
        model.best_epoch_valid_result = best_result
        context.commit('forceUpdateModelList')
        context.commit('forceUpdatePredictionPage')
      }
    }, error_handler_creator(context))
  },

  async runPredictionThread (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/thread/prediction/run/' + model_id
    return axios.get(url)
      .then(function (response) {
        let model = context.getters.getModelById(model_id)
        model.state = STATE.PRED_CREATED // TODO: Remove this line.
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
  },

  async pollingPrediction (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/polling/prediction/model/' + model_id
    const request_source = {'prediction': model_id}
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
        let r = response.data
        let state = r.state
        let need_pull = response.data.need_pull

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
          context.dispatch('loadPredictionResult', model_id)
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },

  async loadPredictionResult (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/load/prediction/result/' + model_id
    return axios.get(url).then(function (response) {
      const model = context.getters.getModelById(model_id)
      if (model) {
        const r = response.data
        const result = r.result
        model.prediction_result = result
        context.commit('forceUpdatePredictionPage')
      }
    }, error_handler_creator(context))
  },

  /*****
   *
   */
  async startWeightDownload (context, payload) {

  },

  /*****
   *
   */
  async pollingWeightDownload (context, payload) {

  },

  /*****
   *
   */
  async stopModelTrain (context, payload) {
    const model_id = payload
    const url = '/api/renom_img/v2/model/stop/' + model_id
    return axios.get(url).then(function (response) {
    }, error_handler_creator(context))
  },

  /*****
   *
   */
  async startAllPolling (context, payload) {
    const model_list = context.state.models.filter(m => m.id >= 0)
    const current_train_requests = context.state.polling_request_jobs.train
    const current_prediction_requests = context.state.polling_request_jobs.prediction

    for (let m of model_list) {
      if (
        m.state === STATE.CREATED ||
        m.state === STATE.RESERVED ||
        m.state === STATE.STARTED
      ) {
        let need_run_request = true
        for (let model_id in current_train_requests) {
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
        for (let model_id in current_prediction_requests) {
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

  /*****
   *
   */
  async createDataset (context, payload) {
    const url = '/api/renom_img/v2/dataset/create'
    const param = new FormData()
    const name = payload.name
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = 'test'
    const test_dataset_id = payload.test_dataset_id

    param.append('name', name)
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', description)
    param.append('test_dataset_id', test_dataset_id)

    const dataset = new Dataset(task_id, name, ratio, description, test_dataset_id)

    context.commit('addDataset', dataset)

    return axios.post(url, param).then(function (response) {
      if (response.status === 204) return
      const id = response.data.id
      const class_map = response.data.class_map
      const valid_data = response.data.valid_data
      dataset.id = id
      dataset.class_map = class_map
      dataset.valid_data = valid_data
    }, error_handler_creator(context))
  },

  /*****
   *
   */
  async createTestDataset (context, payload) {
    const url = '/api/renom_img/v2/test_dataset/create'
    const name = payload.name
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = 'test'

    const test_dataset = new TestDataset(task_id, name, ratio, description)

    const param = new FormData()
    param.append('name', name)
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', description)

    context.commit('addTestDataset', test_dataset)
    return axios.post(url, param).then(function (response) {
      if (response.status === 204) return
      const id = response.data.id
      const class_map = response.data.class_map
      const test_data = response.data.test_data
      test_dataset.id = id
      test_dataset.class_map = class_map
      test_dataset.test_data = test_data
    }, error_handler_creator(context))
  },
  async confirmDataset (context, payload) {

  },
  async confirmTestDataset (context, payload) {
    const url = '/api/renom_img/v2/dataset/confirm'
    const name = encodeURIComponent(payload.name)
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = encodeURIComponent(payload.description)
    const param = new FormData()
    param.append('name', name)
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', description)
    return axios.post(url, param).then(function (response) {
      if (response.status === 204) return
      console.log([response.data])
      const class_info = response.data
      context.commit('setConfirmTestDataset', class_info)
    }, error_handler_creator(context))
  },
  async loadSegmentationTargetArray (context, payload) {
    let url = '/target/segmentation/' + payload
    return axios.get(url)
  },
  async deployModel (context, payload) {
    let model = payload
    let url = '/api/renom_img/v2/model/deploy/' + model.id
    this.commit('setDeployedModel', model)
    return axios.get(url).then(() => {

    }, error_handler_creator(context))
  },
  async loadDeployedModel (context, payload) {
    let task_id = context.getters.getCurrentTask
    let url = '/api/renom_img/v2/model/load/deployed/task/' + task_id
    return axios.get(url).then((response) => {
      const id = response.data.deployed_id
      if (id) {
        const model = context.getters.getModelById(id)
        this.commit('setDeployedModel', model)
        this.dispatch('loadPredictionResult', model.id)
      }
    }, error_handler_creator(context))
  }
}
