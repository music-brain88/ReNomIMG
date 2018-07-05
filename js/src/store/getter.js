export default {
  /*
  global
  */
  getModelFromId (state) {
    return function (model_id) {
      for (let index in state.models) {
        if (state.models[index].model_id === model_id) {
          return state.models[index]
        }
      }
      return undefined
    }
  },
  getPredictModel (state, getters) {
    if (state.project) {
      return getters.getModelFromId(state.project.deploy_model_id)
    } else {
      return undefined
    }
  },
  getSelectedModel (state, getters) {
    return getters.getModelFromId(state.selected_model_id)
  },
  /*
  dashboard
  */
  getModelsFromState (state) {
    return function (state_id) {
      let ret = []
      for (let index in state.models) {
        if (state.models[index].state === state_id) {
          ret.unshift(state.models[index])
        }
      }
      return ret
    }
  },
  /*
  Detail
  */
  getDatasetName (state) {
    return function (dataset_def_id) {
      if (state.dataset_defs.length !== 0) {
        return state.dataset_defs.filter(d => d.id === dataset_def_id)[0].name
      }
      return ''
    }
  },
  /*
  model samples, prediction sample
  */
  getBBoxCoordinate (state, getters) {
    return function (class_label, box) {
      let w = box[2] * 100
      let h = box[3] * 100
      let x = box[0] * 100 - w / 2
      let y = box[1] * 100 - h / 2
      x = Math.min(Math.max(x, 0), 100)
      y = Math.min(Math.max(y, 0), 100)

      if (x + w > 100) w = 100 - x
      if (y + h > 100) h = 100 - y

      return [class_label, x, y, w, h]
    }
  },

  currentModel (state) {
    return state.models.find((m) => (m.model_id === state.selected_model_id))
  },

  currentDataset (state, getters) {
    const model = getters.currentModel
    if (!model) {
      return
    }

    const dataset_def_id = model.dataset_def_id
    return state.dataset_defs.find((d) => (d.id === dataset_def_id))
  },

  getLastValidationResults (state, getters) {
    let model
    for (let m of state.models) {
      if (m.model_id === state.selected_model_id) {
        model = m
      }
    }
    if (!model) return
    const result = model.best_epoch_validation_result
    const dataset_def_id = model.dataset_def_id
    let dataset = null

    for (let index in state.dataset_defs) {
      if (state.dataset_defs[index].id === dataset_def_id) {
        dataset = state.dataset_defs[index]
      }
    }
    if (!dataset) return

    const path = dataset.valid_imgs
    const label_list = result
    let ret = []
    for (let i = 0; i < path.length; i++) {
      let bboxes = []
      if (label_list && label_list.length > 0) {
        for (let j = 0; j < label_list[i].length; j++) {
          const class_label = label_list[i][j].class
          const box = label_list[i][j].box
          bboxes.push(getters.getBBoxCoordinate(class_label, box))
        }
      }
      ret.push({
        'path': path[i].filename,
        'width': path[i].width,
        'height': path[i].height,
        'predicted_bboxes': bboxes
      })
    }
    return ret
  },
  getPredictResults (state, getters) {
    let result = state.predict_results
    let image_path = result.bbox_path_list
    let label_list = result.bbox_list
    let ret = []

    const i_start = state.predict_page * state.predict_page_image_count
    let i_end = i_start + state.predict_page_image_count
    if (Array.isArray(image_path) && i_end > image_path.length) {
      i_end = image_path.length
    }

    for (let i = i_start; i < i_end; i++) {
      let bboxes = []
      if (label_list && Array.isArray(label_list) && Array.isArray(label_list[0]) && label_list.length > 0) {
        if (Array.isArray(label_list[i])) {
          for (let j = 0; j < label_list[i].length; j++) {
            let class_label = label_list[i][j].class
            let box = label_list[i][j].box
            bboxes.push(getters.getBBoxCoordinate(class_label, box))
          }
        }
        ret.push({
          'path': image_path[i],
          'predicted_bboxes': bboxes
        })
      }
    }
    return ret
  },
  getPageMax (state) {
    if (state.predict_results) {
      return Math.floor(state.predict_results.bbox_path_list.length / state.predict_page_image_count)
    }
  }
}
