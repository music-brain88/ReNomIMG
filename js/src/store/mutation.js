import Project from './classes/project'
import Model from './classes/model'

export default {
  /**
   * Set header page name
   *
   * @param {String} payload.page_name : String which is put on header.
   *
   */
  setPageName (state, payload) {
    state.page_name = payload.page_name
  },

  /**
   * Set project data
   * This function is always called when app stars.
   * This creates new project if there is no project.
   *
   * This function used for changing deploying model.
   *
   * @param {Integer} payload.project_id : Id of project.
   * @param {String} payload.project_name : Name of project.
   * @param {String} payload.project_comment : Comment of project.
   * @param {Integer} payload.deploy_model_id : Deployed model id.
   *
   */
  setProject (state, payload) {
    if (!state.project || state.project.project_id !== payload.project_id) {
      const project = new Project(payload.project_id, payload.project_name, payload.project_comment)
      state.project = project
    }
    state.project.deploy_model_id = payload.deploy_model_id
  },

  /**
   * Set model data.
   * This resets the variable 'state.models'. It will cause screen update.
   *
   * @param {Array} payload.models : Array of model.
   *
   */
  setModels (state, payload) {
    state.models = []
    for (let index in payload.models) {
      // 'Deleted model'(=3) is removed.
      if (payload.models[index].state !== 3) {
        state.models.push(payload.models[index])
      }
    }
  },

  /**
   * Add newly created model to state.models. The model's state should be 'Created'.
   * Newly created model will be registered in server side during this method is running.
   *
   * This function will recreate 'state.models'. It will cause screen update.
   *
   * @param {Integer} payload.project_id
   * @param {Integer} payload.model_id
   * @param {Integer} payload.dataset_def_id
   * @param {Object} payload.hyper_parameters
   * @param {Integer} payload.algorithm
   * @param {Object} payload.algorithm_params
   * @param {Integer} payload.state
   * @param {Integer} payload.total_batch
   * @param {Integer} payload.last_train_loss
   * @param {Integer} payload.last_epoch
   * @param {Integer} payload.last_batch
   * @param {Integer} payload.running_state
   * @param {Array} payload.best_epoch_validation_result
   */
  addModelTemporarily (state, payload) {
    let d = payload
    let m = new Model(d.model_id, d.project_id, d.dataset_def_id,
      d.hyper_parameters, d.algorithm, d.algorithm_params, d.state,
      d.best_epoch_validation_result, d.last_epoch, d.last_batch,
      d.total_batch, d.last_train_loss, d.running_state)
    state.models = [m, ...state.models]
  },

  /**
   * Changes model's state.
   *
   * @param {Object} payload : Array of Model object.
   *  The model object has attributes {'model_id': {'running_state':(Int), 'state':(Int)}}
   *
   */
  updateModelsState (state, payload) {
    for (let index in state['models']) {
      let model = state['models'][index]
      let new_state = payload[parseInt(model.model_id)]
      if (new_state !== undefined) {
        if ('running_state' in new_state && model.running_state && model.running_state !== new_state['running_state']) {
          model.running_state = new_state['running_state']
        }
        if ('state' in new_state && model.state !== new_state['state']) {
          model.state = new_state['state']
        }
      } else {
        model.state = 3
      }
    }
  },

  /**
   * Update model's training progress bar.
   * If one epoch have finished, validation result will be given.
   * Then the model list item will be updated.
   *
   * @param {Integer} payload.model_id : Id of model whose training state will be updated.
   * @param {Integer} payload.total_batch : Loop size of one epoch.
   * @param {Integer} payload.last_batch : Current batch index.
   * @param {Integer} payload.last_epoch : Current epoch index.
   * @param {Integer} payload.running_state : Running state.
   *   This represents training, validating, stopping or starting.
   * @param {Float} payload.batch_loss : Loss that calculated with last batch.
   *
   * Following params are sometimes given as empty.
   *
   * @param {Array} payload.train_loss_list : Average train loss of each epoch.
   * @param {Array} payload.validation_loss_list : Average valid loss of each epoch.
   * @param {Integer} payload.best_epoch : Index of epoch which has minimum valid loss.
   * @param {Float} payload.best_epoch_iou : Iou of valid data at best epoch.
   * @param {Float} payload.best_epoch_map : MAP of valid data at best epoch.
   * @param {Float} payload.best_epoch_validation_result :
   *
   */
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
      state.models = [...state.models] // Update display.
    }
  },

  /**
   * Set the navigation bar show flag.
   *
   * @param {Boolean} payload.flag : If this is true, navigation bar will be shown.
   *
   */
  setNavigationBarShowFlag (state, payload) {
    state.navigation_bar_shown_flag = payload.flag
  },

  /**
   *  Set the alert dialog show flag.
   *
   * @param {Boolean} payload.flag : If this is true, alert modal will be shown.
   *
   */
  setAlertModalFlag (state, payload) {
    state.alert_modal_flag = payload.flag
  },

  /**
   *  Set the error message which will be show as alert modal.
   *
   * @param {String} payload.error_msg : A message shown as alert modal.
   *
   */
  setErrorMsg (state, payload) {
    state.error_msg = payload.error_msg
  },

  /**
   *  Set the add model modal show flag.
   *  'Add model modal' is the modal for setting training hyper parameters.
   *
   * @param {Boolean} payload.add_model_modal_show_flag : Flag.
   *
   */
  setAddModelModalShowFlag (state, payload) {
    state.add_model_modal_show_flag = payload.add_model_modal_show_flag
  },

  /**
   *  Sort model list. The order of 'state.models' effects the order of displayed model list.
   *
   * @param {Integer} payload.sort_by : Integer value which represents order keys.
   *
   *    Model ID: 0
   *    IOU: 1
   *    mAP: 2
   *    Validation loss: 3
   *
   */
  sortModels (state, payload) {
    const sort_columns = ['model_id', 'best_epoch_iou', 'best_epoch_map', 'best_epoch_validation_loss']
    const sort_by = payload.sort_by
    const sort_column = sort_columns[sort_by]
    if (sort_by === 0) {
      // Sort by model ID. This will sort it to descending order.
      state.models.sort(function (a, b) {
        return (a[sort_column] < b[sort_column]) ? 1 : -1
      })
    } else if (sort_by === 1 || sort_by === 2) {
      // Sort by map or iou. This will sort it to descending order.
      state.models.sort(function (a, b) {
        if (typeof (a.best_epoch) === 'undefined') return 1
        if (typeof (b.best_epoch) === 'undefined') return -1
        return (a[sort_column] < b[sort_column]) ? 1 : -1
      })
    } else if (sort_by === 3) {
      // Sort by validation loss. This will sort it to ascending order.
      state.models.sort(function (a, b) {
        if (typeof (a.best_epoch) === 'undefined') return 1
        if (typeof (b.best_epoch) === 'undefined') return -1
        return (a.validation_loss_list[a.best_epoch] < b.validation_loss_list[b.best_epoch]) ? -1 : 1
      })
    }
  },

  /**
   * Change selecting model.
   *
   * @param {Integer} payload.model_id : Id of selecting model.
   *
   */
  setSelectedModel (state, payload) {
    state.selected_model_id = payload.model_id
  },

  /**
   * Change deployed model id.
   *
   * @param {Integer} payload.model_id : Id of selecting model.
   *
   */
  setDeployModelId (state, payload) {
    if (state.project) {
      state.project.deploy_model_id = payload.model_id
    }
  },

  /**
   * Set class name list to state.
   * **This function is no longer used from >= beta0.8.
   *
   * @param {Array} payload.class_names : Array of class name.
   *
   */
  setDatasetInfov0 (state, payload) {
    state.class_names = payload.class_names
  },

  /**
   * Set image modal show flag.
   *
   * @param {Array} payload.flag : If this is true, modal will be shown.
   *
   */
  setImageModalShowFlag (state, payload) {
    state.image_modal_show_flag = payload.flag
  },

  /**
   * Set index of image for show image with modal.
   *
   * @param {Integer} payload.index : Index of image.
   *
   */
  setImageIndexOnModal (state, payload) {
    state.image_index_on_modal = payload.index
  },

  /**
   * Set current page number of prediction sample.
   *
   * @param {Integer} payload.page : The number of current page.
   *
   */
  setValidationPage (state, payload) {
    state.validation_page = payload.page
  },

  /**
   * This set page number. The number will be calculated by image index.
   * If the modal is shown and image index changes, the page number will
   * change according new image index.
   *
   * @param {Object} payload.modal : Current selected model.
   * @param {Integer} payload.img_idx : Index of image.
   *
   */
  setShowModalImageSample (state, payload) {
    state.show_modal_image_sample = payload.modal
    state.idx_active_image_sample = payload.img_idx

    if (state.show_modal_image_sample) {
      const model = state.models.find((m) => (m.model_id === state.selected_model_id))
      const dataset_def_id = model.dataset_def_id
      const dataset = state.dataset_defs.find((d) => (d.id === dataset_def_id))

      const page = dataset.pages.findIndex((r) => payload.img_idx < r[1])
      if (page === -1) {
        state.validation_page = 0
      } else {
        state.validation_page = page
      }
    }
  },

  /**
   * Set result of prediction.
   *
   * @param {Object} payload.predict_results :
   * @param {Object} payload.csv :
   *
   */
  setPredictResult (state, payload) {
    state.predict_running_flag = false
    state.predict_results = payload.predict_results
    state.csv = payload.csv
  },

  /**
   * Set the predict page
   *
   * @param {Object} payload.flag :
   *
   */
  setPredictInfo (state, payload) {
    state.predict_total_batch = payload.predict_total_batch
    state.predict_last_batch = payload.predict_last_batch
  },

  /**
   * Set the prediction progress to the state.
   *
   * @param {Object} payload.predict_page_image_count :
   *
   */
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

  /**
   * Set the flag which represents if the prediction progress modal is shown.
   *
   * @param {Object} payload.flag :
   *
   */
  setPredictRunningFlag (state, payload) {
    state.predict_running_flag = payload.flag
  },

  /**
   * This flushes result of prediction page.
   *
   */
  resetPredictResult (state, payload) {
    state.predict_results = {'bbox_list': [], 'bbox_path_list': []}
  },

  /**
   * Set the flag that represents weight existence.
   *
   * @param {Object} payload.weight_exists :
   *
   */
  setWeightExists (state, payload) {
    state.weight_exists = payload.weight_exists
  },

  /**
   * Set flag that represents show the modal.
   *
   * @param {Object} payload.weight_downloading_modal :
   *
   */
  setWeightDownloadModal (state, payload) {
    state.weight_downloading_modal = payload.weight_downloading_modal
  },

  /**
   * Set progress of weight download state.
   *
   * @param {Object} payload.progress:
   *
   */
  setWeightDownloadProgress (state, payload) {
    state.weight_downloading_progress = Math.round(payload.progress * 10) / 10
  },

  /**
   * Set dataset to state.
   *
   * @param {Object} payload.dataset_defs:
   *
   */
  setDatasetDefs (state, payload) {
    state.dataset_defs = payload.dataset_defs

    const IMG_ROW_HEIGHT = 160
    const IMG_ROW_WIDTH = (1280 - // width
                           12 * 2 - // padding of container
                           72 * 2 - // margin of detection-page
                           216 - // width of tag-list
                           24 - // margin of tag-list
                           4 - // margin
                           4 // image imargin

    )
    const IMG_MARGIN = 4
    for (const dataset of state.dataset_defs) {
      dataset.pages = []

      let nrow = 1
      let curwidth = 0
      let pagefrom = 0
      let rowto = 0

      for (const img of dataset.valid_imgs) {
        const imgwidth = (IMG_ROW_HEIGHT / img.height) * img.width
        if ((curwidth + imgwidth + IMG_MARGIN * 2) >= IMG_ROW_WIDTH) {
          if ((nrow % 3) === 0) {
            dataset.pages.push([pagefrom, rowto])
            pagefrom = rowto
          }
          curwidth = 0
          nrow += 1
        }
        rowto += 1
        curwidth += imgwidth + IMG_MARGIN * 2
      }
      if (pagefrom !== (dataset.valid_imgs.length)) {
        dataset.pages.push([pagefrom, rowto])
      }
    }
  }
}
