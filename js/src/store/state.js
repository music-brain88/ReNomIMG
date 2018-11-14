import { SORTBY, TASK_ID, PAGE_ID } from '@/const.js'

export default {

  polling_request_jobs: { // axios obj here.
    weight_download: [],
    train: [],
    validation: []
  },

  current_task: TASK_ID.DETECTION,
  current_page: PAGE_ID.TRAIN,

  // Models
  models: [],
  selected_model: {},

  // Datasets
  datasets: [],

  // TestDatasets
  test_datasets: [],

  // Filter List
  filters: [],

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
    add_filter: false,
  },

  // Variable of image list pager.
  nth_image_page: 0,
  total_image_page: 0,
}
