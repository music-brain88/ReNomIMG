import Model from './classes/model'
import {getKeyByValue, TASK_ID, SORTBY, getKeyByValueIncludes} from '@/const.js'

export default {
  setAlertModalFlag (state, payload) {
    state.show_alert_modal = payload
  },
  showAlert (state, payload) {
    state.error_msg = payload.show
    state.show_alert_modal = payload.msg
  },
  setCurrentTask (state, payload) {
    const task = payload
    if (task in Object.values(TASK_ID)) {
      state.current_task = task
    } else {
      throw new Error('Not supported task.')
    }
  },
  addModel (state, payload) {
    state.models = [payload, ...state.models]
  },
  showSlideMenu (state, payload) {
    state.show_slide_menu = payload
  },
  setSortOrder (state, payload) {
    let task = state.current_task // Need access through getter.
    if (task in Object.values(TASK_ID)) {
      let value = payload.target.value
      let task_key = getKeyByValue(TASK_ID, task)
      let key = getKeyByValueIncludes(SORTBY[task_key], value)
      state.sort_order = SORTBY[task_key][key].id
    } else {
      throw new Error('Not supported task.')
    }
  }
}
