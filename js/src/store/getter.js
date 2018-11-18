import { STATE, PAGE_ID, ALGORITHM, SORTBY, TASK_ID, getKeyByValue, getKeyByValueIncludes } from '@/const.js'

export default {
  /**
   *
   */
  getRunningModelList (state, getters) {
    return getters.getFilteredModelList.filter(m => m.state === STATE.STARTED)
  },
  getFilteredAndGroupedModelList (state, getters) {
    return [[0.60, 0.30], [0.29, 0.3]]
  },
  getFilterList (state, getters) {
    return [1, 2, 3]
  },
  getFilteredModelList (state, getters) {
    // TODO: Sort by state and task.
    return state.models.filter(m => m.task_id === getters.getCurrentTask)
  },
  getFilteredDatasetList (state, getters) {
    // TODO: Sort by task.
    return state.datasets.filter(d => d.task_id === getters.getCurrentTask)
  },
  getModelById (state, getters) {
    return function (id) {
      let model = state.models.find(m => m.id === id)
      return model
    }
  },
  getCurrentTask (state, getters) {
    return state.current_task
  },
  getSelectedModel (state, getters) {
    return state.selected_model[getters.getCurrentTask]
  },
  getCurrentTaskTitle (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Classification'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'Detection'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'Segmentation'
    }
  },
  getCurrentPageTitle (state, getters) {
    if (state.current_page === PAGE_ID.TRAIN) {
      return 'Train'
    } else if (state.current_page === PAGE_ID.PREDICT) {
      return 'Predict'
    } else if (state.current_page === PAGE_ID.DATASET) {
      return 'Dataset'
    } else if (state.current_page === PAGE_ID.DEBUG) {
      return 'Debug'
    }
  },
  getShowSlideMenu (state, getters) {
    return state.show_slide_menu
  },
  getSortTitle (state, getters) {
    let task = getters.getCurrentTask
    if (task in Object.values(TASK_ID)) {
      let key = getKeyByValue(TASK_ID, task)
      return Object.values(SORTBY[key]).map((item) => { return item.title })
    } else {
      throw new Error('Not supported task.')
    }
  },
  getAlgorithmList (state, getters) {
    let task = getters.getCurrentTask
    if (task in Object.values(TASK_ID)) {
      let key = getKeyByValue(TASK_ID, task)
      return Object.values(ALGORITHM[key]).map((item) => { return item.title })
    } else {
      throw new Error('Not supported task.')
    }
  },
  getAlgorithmIdFromTitle (state, getters) {
    return function (algorithm_title) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        return ALGORITHM[task_key][key].id
      } else {
        throw new Error(algorithm_title + ' is not supported task.')
      }
    }
  },
  getAlgorithmTitleFromId (state, getters) {
    return function (algorithm_id) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_id)
        return ALGORITHM[task_key][key].title
      } else {
        throw new Error(algorithm_id + 'is not supported id.')
      }
    }
  },
  getAlgorithmClassFromId (state, getters) {
    return function (algorithm_id) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_id)
        return ALGORITHM[task_key][key].key
      } else {
        throw new Error(algorithm_id + 'is not supported id.')
      }
    }
  },

  getAlgorithmParamList (state, getters) {
    return function (algorithm_title) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        let alg = ALGORITHM[task_key][key]
        if (alg && alg.params) {
          return alg.params
        } else {
          return []
        }
      } else {
        throw new Error(algorithm_title + 'is not supported title.')
      }
    }
  },
  getOptimizedValidImagesInCurrentPage (state, getters) {
    const model = getSelectedModel()
    const dataset = state.dataset.filter(d => d.id === model.dataset_id)
  }
}
