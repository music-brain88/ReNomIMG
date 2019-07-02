import Model from './classes/model'
import axios from 'axios'
import { STATE } from '@/const.js'
import { Dataset } from './classes/dataset'

function error_handler_creator (context, callback = undefined) {
  return function (error) {
    const status = error.response.status
    if ([200, 400, 403, 404, 405, 500, 503].includes(status)) {
      if (error.response.data || error.response.data.error) {
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
    // TODO: context.dispatch('loadTestDatasetsOfCurrentTask')
    await context.dispatch('loadModelsOfCurrentTask', 'all')
    // await context.dispatch('loadModelsOfCurrentTask', 'running')// TODO: 必要？タイミングは？
    // await context.dispatch('loadModelsOfCurrentTask', 'deployed')// TODO: 必要？タイミングは？
    await context.dispatch('loadDeployedModel')
    // TODO: await context.dispatch('loadDeployedModel')
    context.commit('showLoadingMask', false)
    context.dispatch('startAllPolling')
  },

  /** ***
   * Get models list (No details)
   */
  async loadModelsOfCurrentTask (context, payload) {
    // TODO: const task = context.getters.getCurrentTask
    // TODO: const url = '/api/renom_img/v2/model/load/task/' + task
    const task_name = context.getters.getCurrentTaskName
    const state = payload
    const url = '/renom_img/v2/api/' + task_name + '/models?state=' + state
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        // const model_list = response.data.model_list  // TODO:名称変更していた。
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
          context.commit('addModel', model)

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

  async loadDeployedModel (context, payload) {
    // const task_id = context.getters.getCurrentTask
    const task_name = context.getters.getCurrentTaskName
    const url = '/renom_img/v2/api/' + task_name + '/models?state=deployed'
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        if (!response.data.models[0]) return
        const m = response.data.models[0]

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

        const deployed_model = context.getters.getModelById(model.id)
        context.commit('setDeployedModel', deployed_model)

        // this.dispatch('forceUpdatePage', deployed_model.id)
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async loadDatasetsOfCurrentTask (context) {
    // TODO: const task_id = context.getters.getCurrentTask
    // TODO: const url = '/api/renom_img/v2/dataset/load/task/' + task_id
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
   *TODO: not using currently

  async loadTestDatasetsOfCurrentTask (context) {
    const task_id = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/test_dataset/load/task/' + task_id
    return axios.get(url)
      .then(function (response) {
        if (response.status === 204) return
        for (const ds of response.data.test_dataset_list) {
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
   */

  /** ***
   *
   */
  async createModel (context, payload) {
    // TODO: const url = '/api/renom_img/v2/model/create'
    const task_name = context.getters.getCurrentTaskName
    const url = '/renom_img/v2/api/' + task_name + '/models'
    const hyper_params = payload.hyper_params
    const algorithm_id = payload.algorithm_id
    const dataset_id = payload.dataset_id
    const task_id = payload.task_id
    // TODO: FormDataは使わなくなったそうです。 const param = new FormData()
    const model = new Model(algorithm_id, task_id, hyper_params, dataset_id)
    context.commit('addModel', model)
    model.state = STATE.CREATED

    // Append params.
    // param.append('hyper_parameters', JSON.stringify(hyper_params))
    // param.append('dataset_id', dataset_id)
    // param.append('task_id', task_id)
    // param.append('algorithm_id', algorithm_id)
    return axios.post(url, {
      hyper_parameters: hyper_params,
      dataset_id: dataset_id,
      task_id: task_id,
      algorithm_id: algorithm_id,
    })
      .then(function (response) {
        if (response.status === 204) return
        const id = response.data.model.id
        model.id = id
        model.state = STATE.RESERVED
        context.dispatch('runTrainThread', id)
      }, error_handler_creator(context))
  },
  /** ***
   *
   */
  async removeModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: const url = '/api/renom_img/v2/model/remove/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model_id
    return axios.delete(url)
      .then(function (response) {
        if (response.status === 204) return
        context.commit('rmModel', model_id)
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async runTrainThread (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: const url = '/api/renom_img/v2/model/thread/run/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/train'
    // const param = new FormData()
    // param.append('model_id', model_id)

    return axios.post(url, {
      model_id: model_id
    })
      .then(function (response) {
        const model = context.getters.getModelById(model_id)
        model.state = STATE.CREATED // TODO: Remove this line.
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
  },

  /** ***
   *
   */
  async pollingTrain (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: console.log('### model_id of 【pollingTrain】:' + model_id)
    // TODO: const url = '/api/renom_img/v2/polling/train/model/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/train?model_id=' + model_id
    const request_source = { 'train': model_id }
    const current_requests = context.state.polling_request_jobs.train

    // If polling for the model is already performed, do nothing.
    // TODO: console.log('### current_requests of 【pollingTrain】:' + current_requests)
    if (current_requests.indexOf(model_id) >= 0) {
      // TODO: console.log('###【pollingTrain】RETURN:')
      return
    }

    // Register request.
    context.commit('addPollingJob', request_source)
    // TODO: console.log('###【pollingTrain】GET START:')
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
        console.log('★★★★★★★★★★r of 【pollingTrain】', r)
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
          context.dispatch('loadModelsOfCurrentTaskDetail', model_id)
          // context.dispatch('forceUpdatePage', model_id)
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },

  async runPredictionThread (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: const url = '/api/renom_img/v2/model/thread/prediction/run/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/prediction'
    // TODO: const param = new FormData()
    // TODO: param.append('model_id', model_id)

    const model = context.getters.getModelById(model_id)
    model.state = STATE.PRED_CREATED // TODO: Remove this line.
    model.total_prediction_batch = 0
    model.nth_prediction_batch = 0
    return axios.post(url, {
      model_id: model_id
    })
      .then(function (response) {
        console.log('response in runpridictThread', response)
        context.dispatch('startAllPolling')
      }, error_handler_creator(context))
  },

  async pollingPrediction (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: const url = '/api/renom_img/v2/polling/prediction/model/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/prediction?model_id=' + model_id
    const request_source = { 'prediction': model_id }
    const current_requests = context.state.polling_request_jobs.prediction
    console.log('here is in pollingPrediction!')
    // If polling for the model is already performed, do nothing.
    if (current_requests.indexOf(model_id) >= 0) {
      return
    }

    // Register request.
    context.commit('addPollingJob', request_source)
    return axios.get(url).then(function (response) {
      // This 'response' can be empty.
      console.log('here in response')
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
	console.log('if there is model')
	console.log('response.data', r)
	console.log('need_pull', need_pull)
        // Update model.
        model.state = r.state
        model.running_state = r.running_state
        model.total_prediction_batch = r.total_batch
        model.nth_prediction_batch = r.nth_batch

        if (state === STATE.STOPPED) {
          console.log('1-A. When State === stopped')
        } else {
          console.log('1-B. else')
          context.dispatch('pollingPrediction', model_id)
        }
        if (need_pull){
          console.log('2. when need_pull === true')
          context.dispatch('loadModelsOfCurrentTaskDetail', model_id)
          context.commit('forceUpdatePredictionPage')
          // context.dispatch('forceUpdatePage', model_id)
        }
      }
    }, error_handler_creator(context, function () {
      // Need to reload Model State.
    }))
  },

  async forceUpdatePage (context, payload) {
    const model_id = payload
    const model = context.getters.getModelById(model_id)
    if (model) {
      context.commit('forceUpdateModelList')
      context.commit('forceUpdatePredictionPage')
    }
  },

  /** ***
   *
   */
  async startWeightDownload (context, payload) {

  },

  /** ***
   *
   */
  async pollingWeightDownload (context, payload) {

  },

  /** ***
   *
   */
  async stopModelTrain (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model_id = payload
    // TODO: const url = '/api/renom_img/v2/model/stop/' + model_id
    const url = '/renom_img/v2/api/' + task_name + '/train'
    // TODO: const param = new FormData()
    // TODO: param.append('model_id', model_id)
    // TODO: deleteの時の値の渡し方がバラバラだがOK？

    return axios.delete(url, {
      model_id: model_id
    }).then(function (response) {

    }, error_handler_creator(context))
  },

  /** ***
   * disern the kind of job (Train or Predict)
   */
  async startAllPolling (context, payload) {
    console.log('### startAllPolling STRAT ###')
    const model_list = context.state.models.filter(m => m.id >= 0)
    const current_train_requests = context.state.polling_request_jobs.train
    const current_prediction_requests = context.state.polling_request_jobs.prediction

    console.log('### model_list of 【startAllPolling】 ###:', model_list)
    // TODO: console.dir(model_list)

    for (const m of model_list) {
      // TODO: console.log('### m.state of 【startAllPolling】 ###:' + m.state)
      // TODO: console.log('### m.id of 【startAllPolling】 ###:' + m.id)
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

        console.log('### need_run_request(pollingTrain) of 【startAllPolling】 ###:' + need_run_request)
        if (need_run_request) {
          console.log('### dispatch【pollingTrain】###')
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
        console.log('### need_run_request(pollingPrediction) of 【startAllPolling】 ###:' + need_run_request)
        if (need_run_request) {
          console.log('### dispatch【pollingPrediction】###')
          context.dispatch('pollingPrediction', m.id)
        }
      }
    }
  },

  /** ***
   * TODO: 「best_epoch_valid_result」は直接値を含めて返ってくる仕様に変更
   */
  // async loadBestValidResult (context, payload) {
  //   const model_id = payload
  //
  //   const model = context.getters.getModelById(model_id)
  //   if (model) {
  //     context.commit('forceUpdateModelList')
  //     context.commit('forceUpdatePredictionPage')
  //   }
  //
  //   TODO: const url = '/api/renom_img/v2/model/load/best/result/' + model_id
  //   return axios.get(url).then(function (response) {
  //     const model = context.getters.getModelById(model_id)
  //     if (model) {
  //       const r = response.data
  //       const best_result = r.best_result
  //       model.best_epoch_valid_result = best_result
  //       context.commit('forceUpdateModelList')
  //       context.commit('forceUpdatePredictionPage')
  //     }
  //   }, error_handler_creator(context))
  // },

  // async loadPredictionResult (context, payload) {
  //   const model_id = payload
  //
  //   const model = context.getters.getModelById(model_id)
  //   if (model) {
  //     context.commit('forceUpdatePredictionPage')
  //   }
  //
  //   TODO: const url = '/api/renom_img/v2/model/load/prediction/result/' + model_id
  //   return axios.get(url).then(function (response) {
  //     const model = context.getters.getModelById(model_id)
  //     if (model) {
  //       const r = response.data
  //       const result = r.result
  //       model.last_prediction_result = result
  //       context.commit('forceUpdatePredictionPage')
  //     }
  //   }, error_handler_creator(context))
  // },

  /** ***
   * PUT the tempDataset : not using in current version v2.2
   */
  async createDataset (context, payload) {
    // TODO: console.log('dataset_id of createDataset' + payload.dataset_id)

    const task_name = context.getters.getCurrentTaskName
    const dataset_id = payload.dataset_id
    const url = '/renom_img/v2/api/' + task_name + '/datasets/' + dataset_id

    // TODO: const url = '/api/renom_img/v2/dataset/create'
    // const param = new FormData()
    // const name = payload.name
    // const hash = payload.hash
    // const ratio = payload.ratio
    // const task_id = context.getters.getCurrentTask
    // const description = payload.description
    // const test_dataset_id = payload.test_dataset_id
    // param.append('name', encodeURIComponent(name))
    // param.append('hash', hash)
    // param.append('ratio', ratio)
    // param.append('task_id', task_id)
    // param.append('description', encodeURIComponent(description))
    // param.append('test_dataset_id', test_dataset_id)
    // return axios.post(url, param)
    return axios.put(url).then(function (response) {
      if (response.status === 204) return
      // TODO: const dataset = context.state.confirming_dataset
      // TODO: dataset.id = response.data.dataset_id
      // TODO: context.commit('addDataset', dataset)
      context.dispatch('loadDatasetsOfCurrentTaskDetail', dataset_id)
    }, error_handler_creator(context))
  },

  /** ***
   * TODO:

  async createTestDataset (context, payload) {
    const url = '/api/renom_img/v2/test_dataset/confirm'
    const name = payload.name
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = 'test'

    const test_dataset = new TestDataset(task_id, name, ratio, description)

    const param = new FormData()
    param.append('name', encodeURIComponent(name))
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', encodeURIComponent(description))

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
  */

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
    // TODO: const param = new FormData()
    // TODO: param.append('name', encodeURIComponent(name))
    // TODO: param.append('hash', hash)
    // TODO: param.append('ratio', ratio)
    // TODO: param.append('task_id', task_id)
    // TODO: param.append('description', encodeURIComponent(description))
    // TODO: param.append('test_dataset_id', test_dataset_id)

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

  /* TODO:
  async confirmTestDataset (context, payload) {
    const url = '/api/renom_img/v2/test_dataset/confirm'
    const name = payload.name
    const ratio = payload.ratio
    const task_id = context.getters.getCurrentTask
    const description = payload.description
    const param = new FormData()
    param.append('name', encodeURIComponent(name))
    param.append('ratio', ratio)
    param.append('task_id', task_id)
    param.append('description', encodeURIComponent(description))
    return axios.post(url, param).then(function (response) {
      if (response.status === 204) return
      const class_info = response.data
      context.commit('setConfirmTestDataset', class_info)
    }, error_handler_creator(context))
  },
  */

  async loadSegmentationTargetArray (context, payload) {
    const name = payload.name
    const width = payload.size.width
    const height = payload.size.height
    const query = 'filename=' + name + '&width=' + width + '&height=' + height
    const url = '/renom_img/v2/api/segmentation/datasets/1/mask?' + query

    const callback = payload.callback
    return axios.get(url).then(response => {
      callback(response)
    }, error_handler_creator(context))
  },
  async deployModel (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    // TODO: const url = '/api/renom_img/v2/model/deploy/' + model.id
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model.id

    this.commit('setDeployedModel', model)
    return axios.put(url, {
      deploy: true
    }).then(function (response) {

    }, error_handler_creator(context))
  },
  async unDeployModel (context, payload) {
    // TODO: const task_id = context.getters.getCurrentTask
    // TODO: const url = '/api/renom_img/v2/model/undeploy/' + task_id
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    const url = '/renom_img/v2/api/' + task_name + '/models/' + model.id
    this.commit('unDeployModel')
    return axios.put(url, {
      deploy: false
    }).then(function (response) {

    }, error_handler_creator(context))
  },

  /* TODO: ↓旧ソース。後で消します。
  async loadDeployedModel (context, payload) {
    const task_id = context.getters.getCurrentTask
    const url = '/api/renom_img/v2/model/load/deployed/task/' + task_id
    return axios.get(url).then((response) => {
      const id = response.data.deployed_id
      if (id) {
        const model = context.getters.getModelById(id)
        this.commit('setDeployedModel', model)
        this.dispatch('loadPredictionResult', model.id)
      }
    }, error_handler_creator(context))
  },
  */

  async downloadPredictionResult (context, payload) {
    const task_name = context.getters.getCurrentTaskName
    const model = payload
    const format = 'csv'// 現状はcsvのみですが、今後増える想定です。
    // const url = '/api/renom_img/v2/model/' + model.id + '/export/'
    const url = '/renom_img/v2/api/' + task_name + '/prediction/result?model_id=' + model.id + '&format=' + format
    window.open(url, '__blank')
  },
}
