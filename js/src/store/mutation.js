import Project from './classes/project'
import Model from './classes/model'

export default {
  // set header page name
  setPageName (state, payload) {
    state.page_name = payload.page_name
  },

  // set project data
  setProject (state, payload) {
    if (!state.project || state.project.project_id !== payload.project_id) {
      const project = new Project(payload.project_id, payload.project_name, payload.project_comment)
      state.project = project
    }
    state.project.deploy_model_id = payload.deploy_model_id
  },

  // set model data
  setModels (state, payload) {
    state.models = []
    for (let index in payload.models) {
      // Deleted model(=3) is removed.
      if (payload.models[index].state !== 3) {
        state.models.push(payload.models[index])
      }
    }
    if (state.selected_model_id === undefined) {
      state.selected_model_id = state.models[0].model_id
    }
  },
  addModelTemporarily (state, payload) {
    let d = payload
    let m = new Model(d.model_id, d.project_id, d.dataset_def_id,
      d.hyper_parameters, d.algorithm, d.algorithm_params, d.state,
      d.best_epoch_validation_result, d.last_epoch, d.last_batch,
      d.total_batch, d.last_train_loss, d.running_state)
    state.models = [m, ...state.models]
  },
  // update model state
  updateModelsState (state, payload) {
    for (let index in state['models']) {
      let d = state['models'][index]
      let p = payload[parseInt(d.model_id)]
      if (p !== undefined) {
        if ('running_state' in p && d.running_state && d.running_state !== p['running_state']) {
          d.running_state = p['running_state']
        }
        if ('state' in p && d.state !== p['state']) {
          d.state = p['state']
        }
      } else {
        d.state = 3
      }
    }
  },
  // update progress
  updateProgress (state, payload) {
    let model_id = payload.model_id
    let current_model
    for (let index in state.models) {
      if (state.models[index].model_id === model_id) {
        current_model = state.models[index]
        break
      }
    }
    if (!current_model) return

    current_model.total_batch = payload.total_batch
    current_model.last_epoch = payload.last_epoch
    current_model.last_batch = payload.last_batch
    current_model.running_state = payload.running_state
    current_model.last_train_loss = payload.batch_loss

    if (payload.train_loss_list.length > 0 &&
        payload.validation_loss_list.length > 0) {
      current_model.train_loss_list = payload.train_loss_list
      current_model.validation_loss_list = payload.validation_loss_list
      current_model.best_epoch = payload.best_epoch
      current_model.best_epoch_iou = payload.best_epoch_iou
      current_model.best_epoch_map = payload.best_epoch_map
      current_model.best_epoch_validation_result = payload.best_epoch_validation_result
    }
  },

  /*
  header
  */
  // show navigation bar
  setNavigationBarShowFlag (state, payload) {
    state.navigation_bar_shown_flag = payload.flag
  },

  /*
  alert modal
  */
  setAlertModalFlag (state, payload) {
    state.alert_modal_flag = payload.flag
  },

  setErrorMsg (state, payload) {
    state.error_msg = payload.error_msg
  },

  /*
  model list area
  */
  // set add model modal
  setAddModelModalShowFlag (state, payload) {
    state.add_model_modal_show_flag = payload.add_model_modal_show_flag
  },

  // sort model array
  sortModels (state, payload) {
    const sort_columns = ['model_id', 'best_epoch_iou', 'best_epoch_map', 'best_epoch_validation_loss']
    const sort_by = payload.sort_by
    const sort_column = sort_columns[sort_by]
    if (sort_by === 0) {
      // Model IDでソート 大きい順
      state.models.sort(function (a, b) {
        return (a[sort_column] < b[sort_column]) ? 1 : -1
      })
    } else if (sort_by === 1 || sort_by === 2) {
      // IoU, mAPでソート 大きい順
      state.models.sort(function (a, b) {
        if (typeof (a.best_epoch) === 'undefined') return 1
        if (typeof (b.best_epoch) === 'undefined') return -1
        return (a[sort_column] < b[sort_column]) ? 1 : -1
      })
    } else if (sort_by === 3) {
      // Validation Lossでソート 小さい順
      state.models.sort(function (a, b) {
        if (typeof (a.best_epoch) === 'undefined') return 1
        if (typeof (b.best_epoch) === 'undefined') return -1
        return (a.validation_loss_list[a.best_epoch] < b.validation_loss_list[b.best_epoch]) ? -1 : 1
      })
    }
  },

  // change selected model
  setSelectedModel (state, payload) {
    state.selected_model_id = payload.model_id
  },

  /*
  model detail
  */
  setDeployModelId (state, payload) {
    if (state.project) {
      state.project.deploy_model_id = payload.model_id
    }
  },

  /*
  tag list
  */
  setDatasetInfov0 (state, payload) {
    state.class_names = payload.class_names
  },

  /*
  model sample
  */
  setImageModalShowFlag (state, payload) {
    state.image_modal_show_flag = payload.flag
  },

  setImageIndexOnModal (state, payload) {
    state.image_index_on_modal = payload.index
  },

  setValidationPage (state, payload) {
    state.validation_page = payload.page
  },

  setShowModalImageSample (state, payload) {
    state.show_modal_image_sample = payload.modal
    state.idx_active_image_sample = payload.img_idx
    if (state.show_modal_image_sample) {
      state.validation_page = Math.floor(payload.img_idx / state.validation_num_img_per_page)
    }
  },

  /*
  prediction page
  */
  setPredictResult (state, payload) {
    state.predict_running_flag = false
    state.predict_results = payload.predict_results
    state.csv = payload.csv
  },
  setPredictInfo (state, payload) {
    state.predict_total_batch = payload.predict_total_batch
    state.predict_last_batch = payload.predict_last_batch
  },
  setPredictPage (state, payload) {
    const max_chunk = Math.floor(state.predict_results.bbox_path_list.length / state.predict_page_image_count)
    if (payload.page > max_chunk) {
      state.predict_page = max_chunk
    } else if (payload.page < 0) {
      state.predict_page = 0
    } else {
      state.predict_page = payload.page
    }
  },

  setPredictRunningFlag (state, payload) {
    state.predict_running_flag = payload.flag
  },

  resetPredictResult (state, payload) {
    state.predict_results = {'bbox_list': [], 'bbox_path_list': []}
  },

  /*
  weight
  */
  setWeightExists (state, payload) {
    state.weight_exists = payload.weight_exists
  },
  setWeightDownloadModal (state, payload) {
    state.weight_downloading_modal = payload.weight_downloading_modal
  },
  setWeightDownloadProgress (state, payload) {
    state.weight_downloading_progress = Math.round(payload.progress * 10) / 10
  },
  setDatasetDefs (state, payload) {
    state.dataset_defs = payload.dataset_defs
  }
}
