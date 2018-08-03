import axios from 'axios'

export default {
  /**
   * This is called when the browser refreshed. This loads model list and dataset according to given project id.
   *
   * @param {Integer} payload.project_id : Current project id.
   *
   */
  async initData (context, payload) {
    await context.dispatch('loadProject', {'project_id': payload.project_id})
    await context.dispatch('loadModels', {'project_id': payload.project_id})
    await context.dispatch('loadDatasetDef', {'project_id': payload.project_id})
  },

  /**
   * This loads project's name and comment.
   *
   * @param {Integer} payload.project_id : Current project id.
   *
   */
  async loadProject (context, payload) {
    const url = '/api/renom_img/v1/projects/' + payload.project_id
    return axios.get(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
          return
        }
        context.commit('setProject', {
          'project_id': response.data.project_id,
          'project_name': response.data.project_name,
          'project_comment': response.data.project_comment,
          'deploy_model_id': response.data.deploy_model_id,
          'gpu_num': response.data.gpu_num
        })
      })
  },

  /**
   * This will load model list according to project_id.
   *
   * @param {Integer} payload.project_id : Current project id.
   *
   */
  async loadModels (context, payload) {
    // This API calls "get_models"
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models'
    return axios.get(url).then(function (response) {
      if (response.data.error_msg) {
        context.commit('setAlertModalFlag', {'flag': true})
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        return
      }
      context.commit('setModels', {'models': response.data})
    })
  },

  /**
   * This will check model's pretrained weight download progress.
   *
   * @param {Integer} payload.i : This is number of progress. If the downloading process done 15%, i should be 1.
   *
   */
  async checkWeightDownloadProgress (context, payload) {
    if (!context.state.weight_exists) {
      let url = '/api/renom_img/v1/weights/progress/' + payload.i
      return axios.get(url)
        .then(function (response) {
          if (response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true})
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
            return
          }
          if (response.data.progress) {
            context.commit('setWeightDownloadModal', {'weight_downloading_modal': true})
            context.commit('setWeightDownloadProgress', {'progress': response.data.progress})
          }
          if (response.data.progress >= 100) {
            context.commit('setWeightExists', {'weight_exists': true})
            context.commit('setWeightDownloadModal', {'weight_downloading_modal': false})
          }
        })
    }
  },

  /**
   * This will creates new model and registers it to database.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.algorithm : Constant of algorithm given by user.
   * @param {Integer} payload.algorithm_params : Algorithm specified parameters given by user.
   * @param {Integer} payload.hyper_parameters : Training hyper parameters given by user.
   * @param {Integer} payload.dataset_def_id : Dataset id given by user.
   *
   */
  // create model before run model
  async createModel (context, payload) {
    // add fd model data
    let fd = new FormData()
    fd.append('dataset_def_id', payload.dataset_def_id)
    fd.append('hyper_parameters', payload.hyper_parameters)
    fd.append('algorithm', payload.algorithm)
    fd.append('algorithm_params', payload.algorithm_params)
    let url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/model/create'
    return axios.post(url, fd)
  },

  /**
   * This will run specified model. This calls the function 'createModel'.
   * And this runs weight existence check function.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.algorithm : Constant number of algorithm given by user.
   * @param {Integer} payload.algorithm_params : Algorithm specified parameters given by user.
   * @param {Integer} payload.hyper_parameters : Training hyper parameters given by user.
   * @param {Integer} payload.dataset_def_id : Dataset id given by user.
   *
   */
  async runModel (context, payload) {
    const dataset_def_id = JSON.stringify(payload.dataset_def_id)
    const hyper_parameters = JSON.stringify(payload.hyper_parameters)
    const algorithm_params = JSON.stringify(payload.algorithm_params)
    const result = await context.dispatch('createModel', {
      'dataset_def_id': dataset_def_id,
      'hyper_parameters': hyper_parameters,
      'algorithm': payload.algorithm,
      'algorithm_params': algorithm_params
    })
    if (result.data.error_msg) {
      context.commit('setAlertModalFlag', {'flag': true})
      context.commit('setErrorMsg', {'error_msg': result.data.error_msg})
      context.dispatch('loadModels', {'project_id': payload.project_id})
      return
    }
    const model_id = result.data.model_id
    context.commit('addModelTemporarily', {
      'model_id': model_id,
      'project_id': context.state.project.project_id,
      'dataset_def_id': payload.dataset_def_id,
      'hyper_parameters': payload.hyper_parameters,
      'algorithm': payload.algorithm,
      'algorithm_params': payload.algorithm_params,
      'state': 0,
      'best_epoch_validation_result': [],
      'last_epoch': '-',
      'last_batch': '-',
      'total_batch': '-',
      'last_train_loss': '-',
      'running_state': 0
    })
    await context.dispatch('updateModelsState')
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + model_id + '/run'
    axios.get(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        }
        context.dispatch('updateModelsState')
      })
    for (let i = 1; i <= 10; i++) {
      await context.dispatch('checkWeightDownloadProgress', {'i': i})
    }
  },

  /**
   * This will delete model. This will remove trained weight and all of its information from database.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.model_id : Model id which will be deleted.
   *
   */
  deleteModel (context, payload) {
    let url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + payload.model_id
    return axios.delete(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        }
        context.dispatch('updateModelsState')
      })
  },

  /**
   * This will stop model training progress. This will remove trained weight and all of its information from database.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.model_id : Model id which will be stopped.
   *
   */
  stopModel (context, payload) {
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + payload.model_id + '/stop'
    axios.get(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        }
        context.dispatch('updateModelsState')
      })
  },

  /**
   * This function checks the model's state. This function will be called recursively.
   *
   * @param {Integer} payload.project_id : Current project id.
   *
   */
  updateModelsState (context, payload) {
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/update/state'
    return axios.get(url, {
      timeout: 10000
    }).then(function (response) {
      context.commit('updateModelsState', response.data)
    })
  },

  /**
   * This function updates training model's progress bar state.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.model_id : Model id whose progress will be checked.
   *
   */
  updateProgress (context, payload) {
    // // Called from model_progress.vue.
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + payload.model_id + '/progress'

    let fd = new FormData()
    let model = context.getters.getModelFromId(payload.model_id)
    fd.append('last_batch', model.last_batch)
    fd.append('last_epoch', model.last_epoch)
    fd.append('running_state', model.running_state)

    return axios.post(url, fd, {
      timeout: 60000
    }).then(function (response) {
      if (response.data.error_msg) {
        context.commit('setAlertModalFlag', {'flag': true})
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        return
      }
      context.commit('updateProgress', {
        'model_id': payload.model_id,
        'total_batch': response.data.total_batch,
        'last_batch': response.data.last_batch,
        'last_epoch': response.data.last_epoch,
        'batch_loss': response.data.batch_loss,
        'running_state': response.data.running_state,

        // Following variables are possible to be empty list.
        // Then update will not be performed.
        'validation_loss_list': response.data.validation_loss_list,
        'train_loss_list': response.data.train_loss_list,
        'best_epoch': response.data.best_epoch,
        'best_epoch_iou': response.data.best_epoch_iou,
        'best_epoch_map': response.data.best_epoch_map,
        'best_epoch_validation_result': response.data.best_epoch_validation_result
      })

      // updata progress if state is not finished or deleted
      if (response.data.state === 1 || response.data.state === 4) { // If model is running
        context.dispatch('updateProgress', {'model_id': payload.model_id})
      } else {
        context.dispatch('updateModelsState')
      }
    }).catch(function (error) {

    })
  },

  /**
   * This function sets the model be deployed.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.model_id : Model id which will be deployed.
   *
   */
  deployModel (context, payload) {
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + payload.model_id + '/deploy'
    axios.get(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
          return
        }

        context.commit('setDeployModelId', {
          'model_id': payload.model_id
        })
      })
  },

  /**
   * This function sets the model be undeployed.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.model_id : Model id which will be undeployed.
   *
   */
  undeployModel (context, payload) {
    const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + payload.model_id + '/undeploy'
    axios.get(url)
      .then(function (response) {
        if (response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true})
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
          return
        }

        context.commit('setDeployModelId', {
          'model_id': undefined
        })
      })
  },

  /**
   * This function runs prediction thread using deployed model.
   *
   * @param {Integer} payload.project_id : Current project id.
   * @param {Integer} payload.deploy_model_id : Model id which will be used for prediction.
   *
   */
  runPrediction (context, payload) {
    if (context.state.project) {
      context.commit('setPredictRunningFlag', {'flag': true})
      const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + context.state.project.deploy_model_id + '/run_prediction'
      axios.get(url)
        .then(function (response) {
          if (response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true})
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
            context.commit('setPredictRunningFlag', {'flag': false})
            return
          }

          context.commit('setPredictResult', {
            'predict_results': response.data.predict_results,
            'csv': response.data.csv
          })
        })
    }
  },

  /**
   * This updates prediction progress.
   * This function can be run without payload params.
   *
   */
  updatePredictionInfo (context, payload) {
    if (context.state.project) {
      const url = '/api/renom_img/v1/projects/' + context.state.project.project_id + '/models/' + context.state.project.deploy_model_id + '/prediction_info'
      axios.get(url)
        .then(function (response) {
          if (response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true})
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
            return
          }

          context.commit('setPredictInfo', {
            'predict_total_batch': response.data.predict_total_batch,
            'predict_last_batch': response.data.predict_last_batch
          })
          if (context.state.predict_running_flag) {
            context.dispatch('updatePredictionInfo')
          }
        }).catch(function (error) {
          if (context.state.predict_running_flag) {
            context.dispatch('upadtePredictionInfo')
          }
        })
    }
  },

  /**
   * This function registers dataset to database.
   *
   * @param {Integer} payload.ratio : Dataset will be divided by this ratio. Train:Valid = ratio:(1-ratio).
   * @param {Integer} payload.name : Dataset name given by user.
   *
   */
  async registerDatasetDef (context, payload) {
    // add fd model data
    let fd = new FormData()
    fd.append('ratio', payload.ratio)
    fd.append('name', payload.name)

    let url = '/api/renom_img/v1/dataset_defs/'

    context.commit('setDatasetCreateModal', {'dataset_creating_modal': true})

    await axios.post(url, fd).then(function (response) {
      if (response.data.error_msg) {
        context.commit('setAlertModalFlag', {'flag': true})
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        context.commit('setDatasetCreateModal', {'dataset_creating_modal': false})
      } else {
        context.dispatch('loadDatasetDef').then(() => {
          context.commit('setDatasetCreateModal', {'dataset_creating_modal': false})
        })
      }
    })
  },

  /**
   * This function loads all of datasets from database.
   *
   */
  async loadDatasetDef (context, payload) {
    let url = '/api/renom_img/v1/dataset_defs'
    return axios.get(url).then(function (response) {
      if (response.data.error_msg) {
        context.commit('setAlertModalFlag', {'flag': true})
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
      } else {
        context.commit('setDatasetDefs', {
          'dataset_defs': response.data.dataset_defs
        })
      }
    })
  },
  /**
   * This function view datasets detail
   *
   */
  async loadDatasetSplitDetail (context, payload) {
    let url = '/api/renom_img/v1/load_dataset_split_detail'

    let fd = new FormData()
    fd.append('ratio', payload.ratio)
    fd.append('name', payload.name)

    console.log('ratio:', payload.ratio)
    console.log('name:', payload.name)

    context.commit('setDatasetCreateModal', {'dataset_creating_modal': true})

    return axios.post(url, fd).then(function (response) {
      if (response.data.error_msg) {
        console.log('error:', response)
        context.commit('setAlertModalFlag', {'flag': true})
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg})
        context.commit('setDatasetCreateModal', {'dataset_creating_modal': false})
      } else {
        // let max_value = Math.max.apply(null, response.data.map(function (o) { return o.class_maps }))
        console.log('action:', response.data.class_maps)
        context.commit('setDataSplitDetail', response.data)
        // context.commit('setMaxDataDetailValue', max_value)
      }
    })
  }
}
