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
  addPollingJob (state, payload) {
    let key = Object.keys(payload)[0]
    let model_id = payload[key]
    state.polling_request_jobs[key] = [...state.polling_request_jobs[key], model_id]
  },
  rmPollingJob (state, payload) {
    let key = Object.keys(payload)[0]
    let model_id = payload[key]
    state.polling_request_jobs[key] = state.polling_request_jobs[key].filter(n => n !== model_id)
  },
  showSlideMenu (state, payload) {
    state.show_slide_menu = payload
  },
  showModal (state, payload) {
    let key = Object.keys(payload)[0]
    for (let k in state.show_modal) {
      if (k === key) {
        state.show_modal[k] = payload[key]
      } else {
        state.show_modal[k] = false
      }
    }
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
