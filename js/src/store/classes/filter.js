import { STATE, RUNNING_STATE } from '@/const.js'

export default class Filter {
  constructor (item, condition = null, threshold = null) {
    this.item = item
    this.condition = condition
    this.threshold = threshold
  }
  filter (model_list) {
    // Perform filter.
    return []
  }
}
