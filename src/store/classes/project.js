import Model from './model.js'

export default class Project {
  constructor(project_id, project_name, project_comment) {
    this.project_id = project_id;
    this.project_name = project_name;
    this.project_comment = project_comment;
    this.deploy_model_id = undefined;

    this.selected_model_id = undefined;
    this.models = undefined;
  }

  createModels(models) {
    this.models = [];
    for(let index in models) {
      let d = models[index];
      this.models.push(new Model(d.model_id, d.project_id, d.hyper_parameters, d.algorithm, d.algorithm_params, d.state));
      if(index == 0) {
        this.selected_model_id = d.model_id;
      }
      if(d.best_epoch) {
        this.models[index].best_epoch = d.best_epoch;
        this.models[index].train_loss_list = d.train_loss_list;
        this.models[index].validation_loss_list = d.validation_loss_list;
        this.models[index].best_epoch_iou = d.best_epoch_iou;
        this.models[index].best_epoch_map = d.best_epoch_map;
      }
      if(d.best_epoch_validation_result) {
        this.models[index].best_epoch_validation_result = d.best_epoch_validation_result;
      }
    }
  }

  updateModels(models) {
    for(let index in models) {
      let d = models[index];
      let inserted_model = this.getModelFromId(d.model_id);

      if(inserted_model){
        if(inserted_model.best_epoch != d.best_epoch) {
          inserted_model.best_epoch = d.best_epoch;
          inserted_model.best_epoch_iou = d.best_epoch_iou;
          inserted_model.best_epoch_map = d.best_epoch_map;
        }
        if(d.best_epoch_validation_result) {
          inserted_model.best_epoch_validation_result = d.best_epoch_validation_result;
        }
        if(inserted_model.train_loss_list.length < d.train_loss_list.length) {
          inserted_model.train_loss_list.splice(0, inserted_model.train_loss_list.length, ...d.train_loss_list);
        }
        if(inserted_model.validation_loss_list.length < d.validation_loss_list.length) {
          inserted_model.validation_loss_list.splice(0, inserted_model.validation_loss_list.length, ...d.validation_loss_list);
        }
        if(inserted_model.state != d.state) {
          inserted_model.state = d.state;
        }
      }else{
        this.createModels(models);
      }
    }
  }

  getModelFromId(model_id) {
    for(let index in this.models) {
      if(this.models[index].model_id == model_id){
        return this.models[index];
      }
    }
    return undefined
  }

  removeModelFromId(model_id) {
    for(let index in this.models) {
      if(this.models[index].model_id == model_id){
        this.models.splice(index, 1);
      }
    }
  }

  sortModels(sort_by) {
    const sort_columns = ["model_id", "iou_value", "map_value", "validation_loss"];
    const sort_column = sort_columns[sort_by];

    if(sort_by == 0 || sort_by == 1 || sort_by == 2){
      // Model IDでソート　大きい順
      this.models.sort(function(a, b) {
        return (a[sort_column] < b[sort_column]) ? 1 : -1;
      });
    }else if(sort_by == 3) {
      // Validation Lossでソート　小さい順
      this.models.sort(function(a, b) {
        if(!a.best_epoch) return -1;
        if(!b.best_epoch) return 1;
        return (a.validation_loss_list[a.best_epoch] < b.validation_loss_list[b.best_epoch]) ? -1 : 1;
      })
    }
  }
}
