import C from '@/const.js'

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
    return state.models
  },
  getCurrentTask (state, getters) {
    return state.current_task
  },
  getShowSlideMenu (state, getters) {
    return state.show_slide_menu
  },
  getModelResultTitle (state, getters) {
    if (getters.getCurrentTask === C.TASK.CLASSIFICATION) {
      return ['Precision', 'Recall', 'F1']
    } else if (getters.getCurrentTask === C.TASK.DETECTION) {
      return ['mAP', 'IOU']
    } else if (getters.getCurrentTask === C.TASK.SEGMENTATION) {
      return ['Precision', 'Recall', 'F1']
    } else {
      throw new Error('Not supported task.')
    }
  },
  getSortTitle (state, getters) {
    if (getters.getCurrentTask === C.TASK.CLASSIFICATION) {
      return ['Valid/Loss', 'Valid/Precision', 'Valid/Recall', 'Valid/F1']
    } else if (getters.getCurrentTask === C.TASK.DETECTION) {
      return ['Loss', 'IOU', 'mAP']
    } else if (getters.getCurrentTask === C.TASK.SEGMENTATION) {
      return ['Loss', 'Precision', 'Recall', 'F1']
    } else {
      throw new Error('Not supported task.')
    }
  }

}
