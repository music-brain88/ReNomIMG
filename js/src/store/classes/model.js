export default class Model {
  constructor(model_id, project_id, hyper_parameters, algorithm, algorithm_params, state, best_epoch_validation_result, last_epoch) {
    this.model_id = model_id;
    this.project_id = project_id;
    this.hyper_parameters = hyper_parameters;
    this.algorithm = algorithm;
    this.algorithm_params = algorithm_params;
    this.state = state;

    this.train_loss_list = [];
    this.validation_loss_list = [];

    this.best_epoch = undefined;
    this.best_epoch_iou = undefined;
    this.best_epoch_map = undefined;
    this.best_epoch_validation_result = best_epoch_validation_result;

    this.last_epoch = last_epoch;

    // running information
    this.last_batch = 0;
    this.last_train_loss = 0;
    this.total_batch = 0;
    this.running_state = 3;
  }
}
