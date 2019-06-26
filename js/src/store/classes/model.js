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

    this.total_prediction_batch = 0
    this.nth_prediction_batch = 0

    this.train_loss_list = []
    this.valid_loss_list = []

    this.best_epoch_valid_result = null

    // CHANEGE muraishi
    this.last_prediction_result = null

    this.model_list = []
  }
  isDeployable () {
    return (this.state !== STATE.STOPPED)
  }
  isStopped () {
    return (this.state === STATE.STOPPED)
  }
  isTraining () {
    return (this.state === STATE.STARTED) && (this.running_state === RUNNING_STATE.TRAINING)
  }
  isValidating () {
    return (this.state === STATE.STARTED) && (this.running_state === RUNNING_STATE.VALIDATING)
  }
  isPredicting () {
    return this.state === STATE.PRED_STARTED || this.running_state === RUNNING_STATE.PREDICTING || this.state === STATE.PRED_CREATED || this.state === STATE.PRED_RESERVED
  }
  isStopping () {
    return (this.state === STATE.STARTED) && (this.running_state === RUNNING_STATE.STOPPING)
  }
  isWeightDownloading () {
    return (this.state === STATE.STARTED) && (this.running_state === RUNNING_STATE.WEIGHT_DOWNLOADING)
  }
  isRunning () {
    return this.isTraining() || this.isStopping() || this.isValidating()
  }
  isCreated () {
    return this.state === STATE.CREATED
  }
  isReserved () {
    return this.state === STATE.RESERVED
  }
  getBestLoss () {
    let loss = null
    if (this.best_epoch_valid_result) {
      if (this.best_epoch_valid_result.loss) {
        loss = this.best_epoch_valid_result.loss
      }
    }
    return loss
  }
  getResultOfMetric1 () {
    let m1 = '-'
    if (this.task_id === TASK_ID.CLASSIFICATION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.recall !== undefined) {
          m1 = this.best_epoch_valid_result.recall.toFixed(2)
        }
      }
      return {
        metric: 'Recall',
        value: m1
      }
    } else if (this.task_id === TASK_ID.DETECTION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.mAP !== undefined) {
          m1 = this.best_epoch_valid_result.mAP.toFixed(2)
        }
      }
      return {
        metric: 'mAP',
        value: m1
      }
    } else if (this.task_id === TASK_ID.SEGMENTATION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.recall !== undefined) {
          m1 = this.best_epoch_valid_result.recall.toFixed(2)
        }
      }
      return {
        metric: 'Recall',
        value: m1
      }
    }
  }
  getResultOfMetric2 () {
    let m2 = '-'
    if (this.task_id === TASK_ID.CLASSIFICATION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.precision !== undefined) {
          m2 = this.best_epoch_valid_result.precision.toFixed(2)
        }
      }
      return {
        metric: 'Precision',
        value: m2
      }
    } else if (this.task_id === TASK_ID.DETECTION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.IOU !== undefined) {
          m2 = this.best_epoch_valid_result.IOU.toFixed(2)
        }
      }
      return {
        metric: 'IOU',
        value: m2
      }
    } else if (this.task_id === TASK_ID.SEGMENTATION) {
      if (this.best_epoch_valid_result) {
        if (this.best_epoch_valid_result.precision !== undefined) {
          m2 = this.best_epoch_valid_result.precision.toFixed(2)
        }
      }
      return {
        metric: 'Precision',
        value: m2
      }
    }
  }
  getValidResult (index) {
    const task = this.task_id
    // TODO muraishi: best_epoch_valid_result.prediction
    const ret = this.best_epoch_valid_result
    if (!ret) return
    const pred = ret.prediction
    if (!pred) return

    if (task === TASK_ID.CLASSIFICATION) {
      return pred[index]
    } else if (task === TASK_ID.DETECTION) {
      return pred[index]
    } else if (task === TASK_ID.SEGMENTATION) {
      return pred[index]
    }
  }
  getPredictionResult (index) {
    const task = this.task_id

    // CHANEGE muraishi
    const ret = this.last_prediction_result
    if (!ret) return
    const pred = ret.prediction
    if (!pred) return

    if (task === TASK_ID.CLASSIFICATION) {
      return pred[index]
    } else if (task === TASK_ID.DETECTION) {
      return pred[index]
    } else if (task === TASK_ID.SEGMENTATION) {
      return pred[index]
    }
  }
}
