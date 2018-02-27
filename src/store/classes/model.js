export default class Model {
  constructor(model_id, project_id, hyper_parameters, algorithm, algorithm_params, state) {
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
    this.best_epoch_validation_result = {};
  }

  getRoundedIoU() {
    if(this.best_epoch != undefined) {
      return Math.round(this.best_epoch_iou*100);
    }else{
      return '-';
    }
  }

  getRoundedMAP() {
    if(this.best_epoch != undefined) {
      return Math.round(this.best_epoch_map*100);
    }else{
      return '-';
    }
  }

  getRoundedValidationLoss() {
    if(this.best_epoch != undefined) {
      return Math.round(this.validation_loss_list[this.best_epoch]*100)/100;
    }else{
      return '-';
    }
  }
}
