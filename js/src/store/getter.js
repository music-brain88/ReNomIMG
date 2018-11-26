import { GROUPBY, STATE, PAGE_ID, ALGORITHM, SORTBY, TASK_ID, getKeyByValue, getKeyByValueIncludes } from '@/const.js'
import Model from './classes/model'

export default {
  /**
   *
   */
  getRunningModelList (state, getters) {
    return getters.getFilteredModelList.filter(m => m.state === STATE.STARTED)
  },
  getFilteredAndGroupedModelList (state, getters) {
    let array = []
    // Task
    const task = getters.getCurrentTask
    // Filtering
    const filtered_list = getters.getFilteredModelList
    console.log(filtered_list)
    // Grouping
    if (state.group_by === GROUPBY.NONE.key) {
      array = filtered_list
    }
    /*
    else if (state.group_by === GROUPBY.ALGORITHM.key) {
      array = filtered_list.reduce((grouped, m) => {
        let exists = false
        for (let g of grouped) {
          if (g.algorithm_id === m.algorithm_id) {
            g.model_list.push(m)
            exists = true
            break
          }
        }
        if(!exists) {
          let new_m = new Model(m.algorithm_id, task, {}, -1)
          new_m.model_list.push(m)
          grouped.push(new_m)
        }
        return grouped
      }, [])
    } else if (state.group_by === GROUPBY.DATASET.key) {
    }
    */
    return array
  },
  getFilterList (state, getters) {
    return [1, 2, 3]
  },
  getFilteredModelList (state, getters) {
    // TODO: Sort by state and task.
    return state.models.filter(m => m.task_id === getters.getCurrentTask)
  },
  getFilteredDatasetList (state, getters) {
    // TODO: Sort by task.
    return state.datasets.filter(d => d.task_id === getters.getCurrentTask)
  },
  getFilteredTestDatasetList (state, getters) {
    // TODO: Sort by task.
    return state.test_datasets.filter(d => d.task_id === getters.getCurrentTask)
  },
  getDatasetFromId (state, getters) {
    // TODO: Sort by task.
    return function (id) {
      return state.datasets.find(d => d.id === id)
    }
  },
  getModelById (state, getters) {
    return function (id) {
      let model = state.models.find(m => m.id === id)
      return model
    }
  },
  getCurrentTask (state, getters) {
    return state.current_task
  },
  getSelectedModel (state, getters) {
    return state.selected_model[getters.getCurrentTask]
  },
  getDeployedModel (state, getters) {
    return state.deployed_model[getters.getCurrentTask]
  },
  getCurrentTaskTitle (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Classification'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'Detection'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'Segmentation'
    }
  },
  getCurrentPageTitle (state, getters) {
    if (state.current_page === PAGE_ID.TRAIN) {
      return 'Train'
    } else if (state.current_page === PAGE_ID.PREDICT) {
      return 'Predict'
    } else if (state.current_page === PAGE_ID.DATASET) {
      return 'Dataset'
    } else if (state.current_page === PAGE_ID.DEBUG) {
      return 'Debug'
    }
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
  getAlgorithmList (state, getters) {
    let task = getters.getCurrentTask
    if (task in Object.values(TASK_ID)) {
      let key = getKeyByValue(TASK_ID, task)
      return Object.values(ALGORITHM[key]).map((item) => { return item.title })
    } else {
      throw new Error('Not supported task.')
    }
  },
  getAlgorithmIdFromTitle (state, getters) {
    return function (algorithm_title) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        return ALGORITHM[task_key][key].id
      } else {
        throw new Error(algorithm_title + ' is not supported task.')
      }
    }
  },
  getAlgorithmTitleFromId (state, getters) {
    return function (algorithm_id) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_id)
        return ALGORITHM[task_key][key].title
      } else {
        throw new Error(algorithm_id + 'is not supported id.')
      }
    }
  },

  /*
   * For checking the color, see unified.css
   *
   */
  getColorClass (state, getters) {
    return function (model) {
      let task = getters.getCurrentTask
      let state = model.state
      let running_state = model.running_state

      if (state === STATE.CREATED) {
        return 'color-created'
      } else if (state === STATE.RESERVED) {
        return 'color-reserved'
      }

      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], model.algorithm_id)
        return 'color-' + Number(ALGORITHM[task_key][key].id) % 10
      } else {
        throw new Error(algorithm_id + 'is not supported id.')
      }
    }
  },
  getAlgorithmParamList (state, getters) {
    return function (algorithm_title) {
      let task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        let task_key = getKeyByValue(TASK_ID, task)
        let key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        let alg = ALGORITHM[task_key][key]
        if (alg && alg.params) {
          return alg.params
        } else {
          return []
        }
      } else {
        throw new Error(algorithm_title + 'is not supported title.')
      }
    }
  },
  getAlgorithmColor (state, getters) {
    return function (n) {
      let color
      switch (n) {
        // YOLOv1
        case 30:
          color = '#903f84'
          break
        // YOLOv2
        case 31:
          color = '#433886'
          break
        // SSD
        case 32:
          color = '#009453'
          break
        default:
          color = 'black'
          break
      }
      return color
      // if (n === 10) return '#903f84'
      // if (n === 11) return '#433886'
    }
  },
  getTagColor (state, getters) {
    return function (n) {
      if (n % 10 === 0) return '#E7009A'
      if (n % 10 === 1) return '#9F13C1'
      if (n % 10 === 2) return '#582396'
      if (n % 10 === 3) return '#0B20C4'
      if (n % 10 === 4) return '#3F9AAF'
      if (n % 10 === 5) return '#14884B'
      if (n % 10 === 6) return '#BBAA19'
      if (n % 10 === 7) return '#FFCC33'
      if (n % 10 === 8) return '#EF8200'
      if (n % 10 === 9) return '#E94C33'
    }
  },
  getGroupTitles (state, getters) {
    return Object.values(GROUPBY)
  },
  getTitleMetric1 (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Recall [%]'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'mAP [%]'
    }
  },
  getTitleMetric2 (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Precision [%]'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'IOU [%]'
    }
  },
  getImagePageOfPredictionSample (state, getters) {
    const task = getters.getCurrentTask
    return state.nth_image_page[task]
  }
}
