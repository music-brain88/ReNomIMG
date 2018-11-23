import { TASK_ID, STATE, RUNNING_STATE } from '@/const.js'

export default class Model {
  constructor (algorithm_id, task_id, hyper_parameters, dataset_id) {
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

    this.best_epoch_valid_result = null

    this.model_list = []
  }
  getResultOfMetric1 () {
    let m1 = '-'
    if (this.task_id === TASK_ID.DETECTION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.mAP !== undefined) {
          m1 = this.best_epoch_valid_result.mAP.toFixed(2)
        }
      }
      return {
        metric: 'mAP',
        value: m1
      }
    }
  }
  getResultOfMetric2 () {
    let m2 = '-'
    if (this.task_id === TASK_ID.DETECTION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.IOU !== undefined) {
          m2 = this.best_epoch_valid_result.IOU.toFixed(2)
        }
      }
      return {
        metric: 'IOU',
        value: m2
      }
    }
  }
  getResultOfMetric3 () {
    let m3 = '-'
    if (this.task_id === TASK_ID.DETECTION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.loss !== undefined) {
          m3 = this.best_epoch_valid_result.loss.toFixed(2)
        }
      }
      return {
        metric: 'Loss',
        value: m3
      }
    }
  }
}
