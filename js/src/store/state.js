import { GROUPBY, SORTBY, TASK_ID, PAGE_ID } from '@/const.js'

export default {

  polling_request_jobs: { // axios obj here.
    weight_download: [],
    train: [],
    validation: [],
    prediction: []
  },

  current_task: TASK_ID.DETECTION,
  current_page: PAGE_ID.TRAIN,

  // Models
  models: [],
  selected_model: {},
  deployed_model: {},

  // Datasets
  datasets: [],

  // confirm Datasets details
  dataset_details: [],

  // TestDatasets
  test_datasets: [],

  // confirm Teset Datasets details
  test_dataset_details: [],

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
    show_image: false,
  },

  // Variable of image list pager.
  nth_image_page: {
    [TASK_ID.CLASSIFICATION]: 0,
    [TASK_ID.DETECTION]: 0,
    [TASK_ID.SEGMENTATION]: 0,
  },

  nth_prediction_image_page: {
    [TASK_ID.CLASSIFICATION]: 0,
    [TASK_ID.DETECTION]: 0,
    [TASK_ID.SEGMENTATION]: 0,
  },

  // Grouping. // Not use in version 2.0
  group_by: GROUPBY.NONE.key,

  // Image for modal
  modal_image: null,
  modal_prediction: null,
  modal_target: null,
}
