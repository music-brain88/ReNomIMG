import {STATE, RUNNING_STATE} from '@/const.js'

export default class Model {
  constructor (id, task, hyper_parameters, dataset_id, parents) {
    // (Integer) ID of Model. This will defined by database.
    this.id = id
    this.dataset_id = dataset_id

    // States.
    this.state = STATE.CREATED
    this.running_state = RUNNING_STATE.STARTING

    this.total_epoch = 0
    this.last_epoch = 0

    this.total_batch = 0
    this.last_batch = 0

    this.loss_list = {
      'train': [],
      'valid': [],
    }

    this.best_epoch = 0
    this.best_valid_result = {}

    this.model_list = {
      'parent': parents
    }
  }
  get parents () {
    return this.model_list.parent
  }
}
