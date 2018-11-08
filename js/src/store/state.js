import { SORTBY, TASK_ID } from '@/const.js'

export default {
  current_task: TASK_ID.DETECTION,

  // Models
  models: [],

  /*
  alert modal
  */
  // show alert message modal or not
  show_alert_modal: false,

  // error msg from server
  error_msg: '',

  // show slide menu.
  show_slide_menu: false,

  // Sort order
  sort_order: SORTBY.CLASSIFICATION.MODEL_ID.id,

  // Modal
  show_modal: {
    add_model: false,
    add_dataset: false,
    add_both: false,
  }
}
