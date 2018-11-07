import { SORTBY, TASK_ID } from '@/const.js'

export default {

  // Header States
  task_title: 'Classification',
  page_title: 'Train',

  current_task: TASK_ID.CLASSIFICATION,

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
  sort_order: SORTBY.CLASSIFICATION.MODEL_ID.id
}
