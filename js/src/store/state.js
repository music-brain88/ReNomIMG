export default {
  /*
  header
  */
  // page name
  page_name: "",

  // show nav bar or not
  navigation_bar_shown_flag: false,

  // project data
  project: undefined,

  // selected model id
  selected_model_id: undefined,

  // models
  models: [],

  /*
  model progress
  */
  model_progress: {},

  /*
  model list
  */
  // show model add modal or not
  add_model_modal_show_flag: false,

  /*
  tag list
  */
  class_names: {},

  /*
  model sample
  */
  // prediction sample of detection page
  validation_page: 0,

  /*
  prediction page
  */
  // predicted result
  predict_results: {"bbox_list": [], "bbox_path_list": []},
  // predicted csv file name
  csv: "",

  // prediction progress
  predict_running_flag: false,
  predict_total_batch: 0,
  predict_last_batch: 0,

  // predict page
  predict_page: 0,
  predict_page_image_count: 20,

  // show image modal flag
  image_modal_show_flag: false,
  // image data on modal
  image_index_on_modal: undefined,

  /*
  weight
  */
  // weight_exists or not on server
  weight_exists: false,

  // weight downloading progress
  weight_downloading_progress: 0,

  // show weight downloading modal
  weight_downloading_modal: false,

}