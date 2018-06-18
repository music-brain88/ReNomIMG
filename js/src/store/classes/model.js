export default class Model {
  constructor (model_id, project_id, dataset_def_id,
    hyper_parameters, algorithm, algorithm_params,
    state, best_epoch_validation_result, last_epoch,
    last_batch, total_batch, last_train_loss, running_state) {
    // (Integer) ID of Model. This will defined by database.
    this.model_id = model_id

    // (Integer) Project ID which this model belongs with.
    this.project_id = project_id

    // (Integer) Dataset ID. Temporarily, "dataset_def" is used.
    this.dataset_def_id = dataset_def_id

    // (Object) Hyperparameter of training.
    // This is common item in any object detection algorithm.
    this.hyper_parameters = hyper_parameters

    // (Integer) Algorithm of object detection.
    this.algorithm = algorithm

    // (Object) Algorithm specific parameters.
    this.algorithm_params = algorithm_params

    // (Integer) Model state. States are defined in js/src/constant.js
    // 'Created':0
    // 'Returned':1
    // 'Finished':2
    // 'Deleted':3
    // 'Reserved':4
    this.state = state

    // (Array) Training loss list.
    this.train_loss_list = []

    // (Array) Validation loss list. This must be same length as 'train_loss_list'.
    this.validation_loss_list = []

    // (Float) Best validation loss.
    this.best_epoch = 0

    // (Float) Iou at the best validation loss.
    this.best_epoch_iou = 0

    // (Float) mAP at the best validation loss.
    this.best_epoch_map = 0

    // (Array) Predicted bounding box at the best validation loss.
    this.best_epoch_validation_result = best_epoch_validation_result

    // (Bool) This is a flag whether updateProgress(action.js) has been run or not.
    this.has_executed_progress_api = false

    // ### Following stores real time informations. ###
    // (Integer) Epoch count for synchronizing display to server.
    this.last_epoch = last_epoch

    // (Integer) Counts nth batch.
    this.last_batch = last_batch

    // (Integer) Total number of batch.
    this.total_batch = total_batch

    // (Integer) Train loss of last batch.
    this.last_train_loss = last_train_loss

    // (Integer) This represents state of training.
    // 'Training': 0
    // 'Validating': 1
    // 'Predicting': 2
    // 'Starting': 3
    // 'Stopping': 4
    this.running_state = running_state
  }
}
