import Model from './model.js'

export default class Project {
  constructor(project_id, project_name, project_comment) {
    this.project_id = project_id;
    this.project_name = project_name;
    this.project_comment = project_comment;
    this.deploy_model_id = undefined;

    this.selected_model_id = undefined;
    this.models = [];
  }

  createModel(model_data) {
    let m = new Model(model_data.model_id, model_data.project_id,
      model_data.hyper_parameters, model_data.algorithm,
      model_data.algorithm_params, model_data.state);
    if(model_data.best_epoch) {
      m.best_epoch = model_data.best_epoch;
      m.train_loss_list = model_data.train_loss_list;
      m.validation_loss_list = model_data.validation_loss_list;
      m.best_epoch_iou = model_data.best_epoch_iou;
      m.best_epoch_map = model_data.best_epoch_map;
      m.best_epoch_validation_result = model_data.best_epoch_validation_result;
    }
    return m;
  }

  createModels(models) {
    this.models = [];
    for(let index in models) {
      let d = models[index];
      let m = this.createModel(d);
      this.models.unshift(m);
      this.selected_model_id = d.model_id;
    }
  }

  updateModels(models) {
    let model_ids = []
    for(let i in models) {
      model_ids.push(models[i].model_id);
    }

    // delete model
    for(let i in this.models) {
      let index = model_ids.indexOf(this.models[i].model_id);
      if(index == -1) {
        this.models.splice(i, 1);
      }
    }

    // append or update model
    for(let index in models) {
      let d = models[index];
      let inserted_model = this.getModelFromId(d.model_id);

      // update
      if(inserted_model){
        inserted_model.state = d.state;

        inserted_model.best_epoch = d.best_epoch;
        inserted_model.best_epoch_iou = d.best_epoch_iou;
        inserted_model.best_epoch_map = d.best_epoch_map;
        inserted_model.best_epoch_validation_result = d.best_epoch_validation_result;

        inserted_model.train_loss_list.splice(0, inserted_model.train_loss_list.length, ...d.train_loss_list);
        inserted_model.validation_loss_list.splice(0, inserted_model.validation_loss_list.length, ...d.validation_loss_list);
        inserted_model.last_epoch = d.validation_loss_list.length;
      }else{
        // append
        let m = this.createModel(models[index]);
        this.models.unshift(m);
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
    const sort_columns = ["model_id", "best_epoch_iou", "best_epoch_map", "best_epoch_validation_loss"];
    const sort_column = sort_columns[sort_by];
    if(sort_by == 0) {
      // Model IDでソート　大きい順
      this.models.sort(function(a, b) {
        return (a[sort_column] < b[sort_column]) ? 1 : -1;
      });
    }else if(sort_by == 1 || sort_by == 2){
      // IoU, mAPでソート　大きい順
      this.models.sort(function(a, b) {
        if(!a.best_epoch) return 1;
        if(!b.best_epoch) return -1;
        return (a[sort_column] < b[sort_column]) ? 1 : -1;
      });
    }else if(sort_by == 3) {
      // Validation Lossでソート　小さい順
      this.models.sort(function(a, b) {
        if(!a.best_epoch) return 1;
        if(!b.best_epoch) return -1;
        return (a.validation_loss_list[a.best_epoch] < b.validation_loss_list[b.best_epoch]) ? -1 : 1;
      })
    }
  }
}
