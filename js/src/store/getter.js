import { ALGORITHM, SORTBY, TASK_ID, getKeyByValue, getKeyByValueIncludes} from '@/const.js'

export default {
  /**
   *
   */
  getRunningModelList (state, getters) {
    return []
  },
  getFilteredAndGroupedModelList (state, getters) {
    return [[0.60, 0.30], [0.29, 0.3]]
  },
  getFilterList (state, getters) {
    return [1, 2, 3]
  },
  getFilteredModelList (state, getters) {
    // TODO: Sort by state and task.
    return state.models
  },
  getCurrentTask (state, getters) {
    return state.current_task
  },
  getCurrentTaskTitle (state, getters) {
    if (state.current_task == TASK_ID.CLASSIFICATION) {
      return 'Classification'
    } else if (state.current_task == TASK_ID.DETECTION) {
      return 'Detection'
    } else if (state.current_task == TASK_ID.SEGMENTATION) {
      return 'Segmentation'
    }
  },
  getCurrentPageTitle (state, getters) {
    return 'Train'
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
        let key = getKeyByValueIncludes(ALGORITHM[key], algorithm_title)
      } else {
        throw new Error('Not supported task.')
      }
    }
  },

  getAlgorithmParamList (state, getters) {
    return function (algotirhm_title) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algotirhm_title)
        let alg = ALGORITHM[task_key][key]
        if (alg && alg.params) {
          return alg.params
        } else {
          return []
        }
      } else {
        throw new Error('Not supported task.')
      }
    }
  }
}
