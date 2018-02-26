export default class Model {
  constructor(model_id, project_id, dataset_id, total_epoch, seed, algorithm, hyper_parameters, state, best_epoch) {
    this.model_id = model_id;
    this.dataset_id = dataset_id;
    this.total_epoch = total_epoch;
    this.seed = seed;
    this.algorithm = algorithm;
    this.hyper_parameters = hyper_parameters;
    this.state = state;
    this.best_epoch = best_epoch;
    this.max_memory_usage = undefined;
    this.max_memory_usage_forward = undefined;
    this.epochs = undefined;
    this.current_learning_info = {};
  }

  getRoundedIoU() {
    if(this.epochs && this.epochs.length > 0 && this.best_epoch != undefined) {
      return Math.round(this.epochs[this.best_epoch].iou_value*100);
    }else{
      return "-"
    }
  }

  getRoundedMAP() {
    if(this.epochs && this.epochs.length > 0 && this.best_epoch != undefined) {
      return Math.round(this.epochs[this.best_epoch].map_value*100);
    }else{
      return "-"
    }
  }

  getRoundedTrainLoss() {
    if(this.epochs && this.epochs.length > 0 && this.best_epoch != undefined) {
      return Math.round(this.epochs[this.best_epoch].train_loss*100)/100;
    }else{
      return "-"
    }
  }

  getRoundedValidationLoss() {
    if(this.epochs && this.epochs.length > 0 && this.best_epoch != undefined) {
      return Math.round(this.epochs[this.best_epoch].validation_loss*100)/100;
    }else{
      return "-"
    }
  }

  getTrainLoss() {
    let ret = [];
    if(this.epochs && this.epochs.length > 0) {
      for(let index in this.epochs) {
        ret.push(Math.round(this.epochs[index].train_loss*1000)/1000);
      }
      return ret;
    }
  }

  getValidationLoss() {
    let ret = [];
    if(this.epochs && this.epochs.length > 0) {
      for(let index in this.epochs) {
        ret.push(Math.round(this.epochs[index].validation_loss*1000)/1000);
      }
      return ret;
    }
  }
}
