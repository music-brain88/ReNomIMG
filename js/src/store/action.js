import Model from './classes/model'
import axios from 'axios'
import { STATE } from '@/const.js'
import { Dataset, TestDataset } from './classes/dataset'

export default {
  /*****
   *
   */
  async init (context, payload) {
    context.dispatch('loadDatasetsOfCurrentTask')
    context.dispatch('loadTestDatasetsOfCurrentTask')
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
        if (response.status === 204) return
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
  async loadDatasetsOfCurrentTask (context) {
    const task_id = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/dataset/load/task/' + task_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
          return
        }
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
      })
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
        let error_msg = response.data.error_msg
        if (error_msg) {
          context.commit('showAlert', {'show': true, 'msg': error_msg})
          return
        }
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
        if (response.status === 204) return
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
      let error_msg = response.data.error_msg
      if (error_msg) {
        context.commit('showAlert', {'show': true, 'msg': error_msg})
        return
      }
      const id = response.data.id
      const class_map = response.data.class_map
      const valid_data = response.data.valid_data
      dataset.id = id
      dataset.class_map = class_map
      dataset.valid_data = valid_data
    })
  },

  /*****
   *
   */
  async createTestDataset (context, payload) {
    const url = '/api/renom_img/v2/test_dataset/create'
    const param = new FormData()
    const name = payload.name
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = 'test'

    param.append('name', name)
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', description)

    const test_dataset = new TestDataset(task_id, name, ratio, description)

    context.commit('addTestDataset', test_dataset)

    return axios.post(url, param).then(function (response) {
      if (response.status === 204) return
      let error_msg = response.data.error_msg
      if (error_msg) {
        context.commit('showAlert', {'show': true, 'msg': error_msg})
        return
      }
      const id = response.data.id
      const class_map = response.data.class_map
      const test_data = response.data.test_data
      test_dataset.id = id
      test_dataset.class_map = class_map
      test_dataset.test_data = test_data
    })
  },

}
