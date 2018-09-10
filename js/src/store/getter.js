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
    var ret = []
    const IMG_ROW_WIDTH = 862 // width
    const IMG_HEIGHT = 180
    const IMG_MARGIN = 6 * 2

    var page_count = 0
    if (state.predict_results.prediction_file_list) {
      var counter = 0
      var summed_width = 0
      for (let i = 0; i < state.predict_results.prediction_file_list.length; i++) {
        let img = state.predict_results.prediction_file_list[i]
        let label = state.predict_results.bbox_list[i]

        let img_width = IMG_HEIGHT / img.height * img.width + IMG_MARGIN

        if (summed_width + img_width < IMG_ROW_WIDTH) {
          summed_width += img_width
        } else {
          summed_width = img_width
          counter += 1
          if (counter === 3) {
            counter = 0
            page_count += 1
          }
        }

        if (state.predict_page === page_count) {
          let boxes = []
          for (let j = 0; j < label.length; j++) {
            boxes.push(getters.getBBoxCoordinate(
              label[j].class,
              label[j].box
            ))
          }
          ret.push({
            'path': img.path,
            'predicted_bboxes': boxes
          })
        } else if (state.predict_page < page_count) {
          break
        }
      }
    }
    return ret
  },
  getPageMax (state) {
    const IMG_ROW_WIDTH = 862 // width
    const IMG_HEIGHT = 180
    const IMG_MARGIN = 6 * 2

    var page_count = 0
    if (state.predict_results.prediction_file_list) {
      var counter = 0
      var summed_width = 0
      for (let img of state.predict_results.prediction_file_list) {
        let img_width = IMG_HEIGHT / img.height * img.width + IMG_MARGIN
        if (summed_width + img_width < IMG_ROW_WIDTH) {
          summed_width += img_width
        } else {
          summed_width = img_width
          counter += 1
          if (counter === 3) {
            counter = 0
            page_count += 1
          }
        }
      }
    }
    return page_count
  }
}
