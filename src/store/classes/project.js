import Model from './model.js'

export default class Project {
  constructor(project_id, project_name, project_comment) {
    this.project_id = project_id;
    this.project_name = project_name;
    this.project_comment = project_comment;

    this.selected_model_id = undefined;
    this.models = undefined;
  }

  createModels(models) {
    this.models = [];
    for(let index in models) {
      let d = models[index];
      this.models.push(new Model(d.model_id, d.project_id, d.dataset_id, d.total_epoch, d.seed, d.algorithm, d.hyper_parameter, d.state, d.best_epoch));
      if(index == 0) {
        this.selected_model_id = d.model_id;
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

    if(sort_by == 0){
      // Model IDでソート　大きい順
      this.models.sort(function(a, b) {
        return (a[sort_column] < b[sort_column]) ? 1 : -1;
      });
    }else if(sort_by == 1 || sort_by == 2) {
      // IoU, mAPでソート　大きい順
      this.models.sort(function(a, b) {
        if(!a.best_epoch) return 1;
        if(!b.best_epoch) return -1;
        return (a.epochs[a.best_epoch][sort_column] < b.epochs[b.best_epoch][sort_column]) ? 1 : -1;
      });
    }else if(sort_by == 3) {
      // Validation Lossでソート　小さい順
      this.models.sort(function(a, b) {
        if(!a.best_epoch) return -1;
        if(!b.best_epoch) return 1;
        return (a.epochs[a.best_epoch][sort_column] < b.epochs[b.best_epoch][sort_column]) ? -1 : 1;
      })
    }
  }
}
