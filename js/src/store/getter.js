import { SORT_DIRECTION, FILTER, GROUPBY, STATE, PAGE_ID, ALGORITHM, SORTBY, TASK_ID, getKeyByValue, getKeyByValueIncludes } from './../const.js'

/* eslint no-unused-vars: 0 */
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

    // Sort
    if (state.sort_order === SORTBY.ID) {
      if (state.sort_order_direction === SORT_DIRECTION.DESCENDING) {
        return array.sort((m1, m2) => (m1.id <= m2.id) ? 1 : -1)
      } else {
        return array.sort((m1, m2) => (m1.id > m2.id) ? 1 : -1)
      }
    } else if (state.sort_order === SORTBY.ALG) {
      if (state.sort_order_direction === SORT_DIRECTION.DESCENDING) {
        return array.sort((m1, m2) => (m1.algorithm_id <= m2.algorithm_id) ? 1 : -1)
      } else {
        return array.sort((m1, m2) => (m1.algorithm_id > m2.algorithm_id) ? 1 : -1)
      }
    } else if (state.sort_order === SORTBY.LOSS) {
      if (state.sort_order_direction === SORT_DIRECTION.ASCENDING) {
        return array.sort((m1, m2) => {
          if (!m1.getBestLoss() && !m2.getBestLoss()) {
            return (m1.id > m2.id) ? 1 : -1
          } else if (m1.getBestLoss() && m2.getBestLoss()) {
            return (m1.getBestLoss() > m2.getBestLoss()) ? 1 : -1
          } else if (m1.getBestLoss()) {
            return -1
          } else if (m2.getBestLoss()) {
            return 1
          }
        })
      } else {
        return array.sort((m1, m2) => {
          if (!m1.getBestLoss() && !m2.getBestLoss()) {
            return (m1.id < m2.id) ? 1 : -1
          } else if (m1.getBestLoss() && m2.getBestLoss()) {
            return (m1.getBestLoss() < m2.getBestLoss()) ? 1 : -1
          } else if (m1.getBestLoss()) {
            return -1
          } else if (m2.getBestLoss()) {
            return 1
          }
        })
      }
    } else if (state.sort_order === SORTBY.M1) {
      if (state.sort_order_direction === SORT_DIRECTION.DESCENDING) {
        return array.sort((m1, m2) => {
          const mm1 = m1.getResultOfMetric1().value
          const mm2 = m2.getResultOfMetric1().value
          if (mm1 === '-' && mm2 === '-') {
            return (m1.id < m2.id) ? 1 : -1
          } else if (mm1 !== '-' && !mm2 !== '-') {
            return (mm1 < mm2) ? 1 : -1
          } else if (mm1 !== '-') {
            return -1
          } else if (mm2 !== '-') {
            return 1
          }
        })
      } else {
        return array.sort((m1, m2) => {
          const mm1 = m1.getResultOfMetric1().value
          const mm2 = m2.getResultOfMetric1().value
          if (mm1 === '-' && mm2 === '-') {
            return (m1.id > m2.id) ? 1 : -1
          } else if (mm1 !== '-' && mm2 !== '-') {
            return (mm1 > mm2) ? 1 : -1
          } else if (mm1 !== '-') {
            return -1
          } else if (mm2 !== '-') {
            return 1
          }
        })
      }
    } else if (state.sort_order === SORTBY.M2) {
      if (state.sort_order_direction === SORT_DIRECTION.DESCENDING) {
        return array.sort((m1, m2) => {
          const mm1 = m1.getResultOfMetric2().value
          const mm2 = m2.getResultOfMetric2().value
          if (mm1 === '-' && mm2 === '-') {
            return (m1.id < m2.id) ? 1 : -1
          } else if (mm1 !== '-' && mm2 !== '-') {
            return (mm1 < mm2) ? 1 : -1
          } else if (mm1 !== '-') {
            return -1
          } else if (mm2 !== '-') {
            return 1
          }
        })
      } else {
        return array.sort((m1, m2) => {
          const mm1 = m1.getResultOfMetric2().value
          const mm2 = m2.getResultOfMetric2().value
          if (mm1 === '-' && mm2 === '-') {
            return (m1.id > m2.id) ? 1 : -1
          } else if (mm1 !== '-' && mm2 !== '-') {
            return (mm1 > mm2) ? 1 : -1
          } else if (mm1 !== '-') {
            return -1
          } else if (mm2 !== '-') {
            return 1
          }
        })
      }
    }
    return array
  },
  getFilterList (state, getters) {
    return state.filters
  },
  getFilterItemsOfCurrentTask (state, getters) {
    let key = ''
    const task = getters.getCurrentTask
    if (task === TASK_ID.CLASSIFICATION) {
      key = 'CLASSIFICATION'
    } else if (task === TASK_ID.DETECTION) {
      key = 'DETECTION'
    } else if (task === TASK_ID.SEGMENTATION) {
      key = 'SEGMENTATION'
    } else {
      return {}
    }
    return FILTER[key]
  },
  getFilteredModelList (state, getters) {
    // TODO: Sort by state and task.
    let filtered = []
    const task_filtered_model_list = state.models.filter(m => m.task_id === getters.getCurrentTask)
    filtered = task_filtered_model_list
    // console.log('【currentModels in getFilteredModelList】')
    // console.log(filtered)
    for (const flt of state.filters) {
      filtered = flt.filter(filtered)
    }
    // console.log('【filtered in getFilteredModelList】')
    // console.log(filtered)
    return filtered
  },
  getFilteredDatasetList (state, getters) {
    // TODO: Sort by task.
    return state.datasets.filter(d => d.task_id === getters.getCurrentTask)
  },
  getFilteredTestDatasetList (state, getters) {
    // TODO: Sort by task.
    return state.test_datasets.filter(d => d.task_id === getters.getCurrentTask)
  },
  getModelById (state, getters) {
    return function (id) {
      const model = state.models.find(m => m.id === id)
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
  /* TODO: エラー原因だったため取り急ぎコメントアウト
  getDatasetDetail (state, getters) {
    return state.dataset_details[0]
  },
  */
  getDatasetFromId (state, getters) {
    return function (id) {
      return state.datasets.find(n => n.id === id)
    }
  },
  /* TODO: エラー原因だったため取り急ぎコメントアウト
  getTestDatasetDetail (state, getters) {
    return state.test_dataset_details[0]
  },
  */
  getCurrentTaskTitle (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Classification'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'Detection'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'Segmentation'
    }
  },
  getCurrentTaskName (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'classification'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'detection'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'segmentation'
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
  /* TODO: エラー原因だったため取り急ぎコメントアウト
  getSortTitle (state, getters) {
    const task = getters.getCurrentTask
    if (task in Object.values(TASK_ID)) {
      const key = getKeyByValue(TASK_ID, task)
      return Object.values(SORTBY[key]).map((item) => { return item.title })
    } else {
      throw new Error('Not supported task.')
    }
  },
  */
  getAlgorithmList (state, getters) {
    const task = getters.getCurrentTask
    if (task in Object.values(TASK_ID)) {
      const key = getKeyByValue(TASK_ID, task)
      return Object.values(ALGORITHM[key]).map((item) => { return item.title })
    } else {
      throw new Error('Not supported task.')
    }
  },
  getAlgorithmIdFromTitle (state, getters) {
    return function (algorithm_title) {
      const task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        const task_key = getKeyByValue(TASK_ID, task)
        const key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        return ALGORITHM[task_key][key].id
      } else {
        throw new Error(algorithm_title + ' is not supported task.')
      }
    }
  },
  getAlgorithmTitleFromId (state, getters) {
    return function (algorithm_id) {
      const task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        const task_key = getKeyByValue(TASK_ID, task)
        const key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_id)
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
      const task = getters.getCurrentTask
      const state = model.state

      if (state === STATE.CREATED) {
        return 'color-created'
      } else if (state === STATE.RESERVED) {
        return 'color-reserved'
      }

      if (task in Object.values(TASK_ID)) {
        const task_key = getKeyByValue(TASK_ID, task)
        const key = getKeyByValueIncludes(ALGORITHM[task_key], model.algorithm_id)
        return 'color-' + Number(ALGORITHM[task_key][key].id) % 10
      } else {
        throw new Error(model.algorithm_id + 'is not supported id.')
      }
    }
  },
  getAlgorithmParamList (state, getters) {
    return function (algorithm_title) {
      const task = getters.getCurrentTask
      if (task in Object.values(TASK_ID)) {
        const task_key = getKeyByValue(TASK_ID, task)
        const key = getKeyByValueIncludes(ALGORITHM[task_key], algorithm_title)
        const alg = ALGORITHM[task_key][key]
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
      switch (n % 10) {
        // judge by first digit
        // if n = 10
        // the color will set case 0 variable
        // this is using in d3
        case 0:
          color = '#903e84'
          break
        case 1:
          color = '#423885'
          break
        case 2:
          color = '#136eab'
          break
        case 3:
          color = '#009453'
          break
        case 4:
          color = '#93b447'
          break
        case 5:
          color = '#dfd238'
          break
        default:
          color = 'black'
          break
      }
      return color
    }
  },
  getGroupTitles (state, getters) {
    return Object.values(GROUPBY)
  },
  getTitleMetric1 (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Recall'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'mAP'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'Recall'
    }
  },
  getTitleMetric2 (state, getters) {
    if (state.current_task === TASK_ID.CLASSIFICATION) {
      return 'Precision'
    } else if (state.current_task === TASK_ID.DETECTION) {
      return 'IOU'
    } else if (state.current_task === TASK_ID.SEGMENTATION) {
      return 'Precision'
    }
  },
  getImagePageOfValid (state, getters) {
    const task = getters.getCurrentTask
    return state.nth_image_page[task]
  },
  getImagePageOfPrediction (state, getters) {
    const task = getters.getCurrentTask
    return state.nth_prediction_image_page[task]
  },
  isTaskClassification (state, getters) {
    return TASK_ID.CLASSIFICATION === getters.getCurrentTask
  },
  isTaskDetection (state, getters) {
    return TASK_ID.DETECTION === getters.getCurrentTask
  },
  isTaskSegmentation (state, getters) {
    return TASK_ID.SEGMENTATION === getters.getCurrentTask
  },
}
