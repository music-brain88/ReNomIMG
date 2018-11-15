import {STATE, RUNNING_STATE} from '@/const.js'

export default class Model {
  constructor (algorithm_id, task_id, hyper_parameters, dataset_id, parents) {
    // (Integer) ID of Model. This will defined by database.
    this.id = -1
    this.dataset_id = dataset_id
    this.task_id = task_id
    this.algorithm_id = algorithm_id
    this.hyper_parameters = hyper_parameters

    // States.
    this.state = STATE.CREATED
    this.running_state = RUNNING_STATE.STARTING

    this.total_epoch = 0
    this.nth_epoch = 0

    this.total_batch = 0
    this.nth_batch = 0
    this.last_batch_loss = 0

    this.train_loss_list = []
    this.valid_loss_list = []

    this.best_epoch_valid_result = {}

    this.model_list = []
  }
}
