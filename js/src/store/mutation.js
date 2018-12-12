import { SORT_DIRECTION, GROUPBY, PAGE_ID, getKeyByValue, TASK_ID, SORTBY, getKeyByValueIncludes } from '@/const.js'

export default {
  resetState (state, payload) {
    state.datasets = []
    state.test_datasets = []
    state.models = []
  },
  showAlert (state, payload) {
    state.show_alert_modal = true
    state.error_msg = payload
  },
  hideAlert (state, payload) {
    state.show_alert_modal = false
    state.error_msg = ''
  },
  setCurrentTask (state, payload) {
    const task = payload
    if (task in Object.values(TASK_ID)) {
      state.current_task = task
    } else {
      throw new Error('Not supported task.')
    }
  },
  setGoupBy (state, payload) {
    let key = payload
    state.group_by = key
  },
  setSelectedModel (state, payload) {
    const task_id = state.current_task
    state.selected_model = Object.assign({...state.selected_model}, {[task_id]: payload})
  },
  setDeployedModel (state, payload) {
    const task_id = state.current_task
    const model_task_id = payload.task_id
    if (task_id === model_task_id) {
      state.deployed_model = Object.assign({...state.deployed_model}, {[task_id]: payload})
    }
  },
  unDeployModel (state, payload) {
    const task_id = state.current_task
    state.deployed_model = Object.assign(...state.deployed_model, {[task_id]: undefined})
  },
  forceUpdateModelList (state, payload) {
    state.models = [...state.models]
  },
  forceUpdatePredictionPageSample (state, payload) {
    const page = state.nth_image_page
    state.nth_image_page = {...page}
  },
  forceUpdatePredictionPage (state, payload) {
    const page = state.nth_prediction_image_page
    state.nth_prediction_image_page = {...page}
  },
  setCurrentPage (state, payload) {
    const page = payload
    if (Object.values(PAGE_ID).find(n => n === page) !== undefined) {
      state.current_page = page
    } else {
      throw new Error('Not supported task.')
    }
  },
  setConfirmingFlag (state, payload) {
    state.confirming_flag = payload
  },
  setConfirmingDataset (state, payload) {
    state.confirming_dataset = payload
  },
  setConfirmingTestDataset (state, payload) {
    state.confirming_test_dataset = payload
  },
  addDataset (state, payload) {
    if (state.datasets.find(n => n.id === payload.id) === undefined) {
      state.datasets = [payload, ...state.datasets]
    }
  },
  addTestDataset (state, payload) {
    if (state.test_datasets.find(n => n.id === payload.id) === undefined) {
      state.test_datasets = [payload, ...state.test_datasets]
    }
  },
  addModel (state, payload) {
    if (state.models.find(n => n.id === payload.id) === undefined) {
      state.models = [payload, ...state.models]
    }
  },
  rmModel (state, payload) {
    if (state.models.find(n => n.id === payload.id) === undefined) {
      state.models = state.models.filter(m => m.id !== payload)
    }
  },
  addFilter (state, payload) {
    if (state.filters.find(f => f == payload) === undefined) {
      state.filters = [...state.filters, payload]
    }
  },
  flushFilter (state, payload) {
    state.filters = []
  },
  rmFilter (state, payload) {
    state.filters = state.filters.filter(f => f !== payload)
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
  showLoadingMask (state, payload) {
    state.show_loading_mask = payload
  },
  setImageModalData (state, payload) {
    state.modal_index = payload
  },
  setSortOrder (state, payload) {
    let task = state.current_task // Need access through getter.
    if (task in Object.values(TASK_ID)) {
      state.sort_order = SORTBY[payload]
    } else {
      throw new Error('Not supported task.')
    }
  },
  toggleSortOrder (state, payload) {
    let task = state.current_task // Need access through getter.
    if (task in Object.values(TASK_ID)) {
      if (state.sort_order_direction === SORT_DIRECTION.DESCENDING) {
        state.sort_order_direction = SORT_DIRECTION.ASCENDING
      } else {
        state.sort_order_direction = SORT_DIRECTION.DESCENDING
      }
    } else {
      throw new Error('Not supported task.')
    }
  },
  setImagePageOfValid (state, payload) {
    const task = state.current_task
    state.nth_image_page[task] = payload
  },
  setImagePageOfPrediction (state, payload) {
    const task = state.current_task
    state.nth_prediction_image_page[task] = payload
  },
  selectNextModel (state, payload) {
    const task = state.current_task
    const mlist = state.models.filter(m => m.task_id === task)
    let current = state.selected_model[task]
    let index = 0
    if (current) {
      index = mlist.indexOf(current) + 1
    }
    if (mlist.length > index) {
      state.selected_model = Object.assign(...state.selected_model, {[task]: mlist[index]})
    }
  },
  selectPrevModel (state, payload) {
    const task = state.current_task
    const mlist = state.models.filter(m => m.task_id === task)
    let current = state.selected_model[task]
    let index = 0
    if (current) {
      index = mlist.indexOf(current) - 1
    }
    if (index >= 0) {
      state.selected_model = Object.assign(...state.selected_model, {[task]: mlist[index]})
    }
  }
}
