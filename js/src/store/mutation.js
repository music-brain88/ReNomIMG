import Model from './classes/model'
import C from '../const.js'

export default {
  setAlertModalFlag (state, payload) {
    state.show_alert_modal = payload
  },
  showAlert (state, payload) {
    state.error_msg = payload.show
    state.show_alert_modal = payload.msg
  },
  setCurrentTask (state, payload) {
    const task = payload.task
    assert(task in Object.values(C.TASK))
    state.current_task = task
  },
  addModel (state, payload) {
    state.models = [payload, ...state.models]
  },
  showSlideMenu (state, payload) {
    state.show_slide_menu = payload
  }
}
