import { SORTBY, TASK_ID, getKeyByValue } from '@/const.js'

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
}
