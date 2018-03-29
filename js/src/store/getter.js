import * as constant from '@/constant'

export default {
  /*
  global
  */
  getModelFromId(state) {
    return function(model_id) {
      for(let index in state.models) {
        if(state.models[index].model_id == model_id){
          return state.models[index];
        }
      }
      return undefined
    }
  },
  getPredictModel(state, getters) {
    if(state.project){
      return getters.getModelFromId(state.project.deploy_model_id);
    }
  },
  getSelectedModel(state, getters) {
    return getters.getModelFromId(state.selected_model_id);
  },
  /*
  dashboard
  */
  getModelsFromState(state) {
    return function(state_id) {
      let ret = [];
      for(let index in state.models) {
        if(state.models[index].state == state_id) {
          ret.unshift(state.models[index]);
        }
      }
      return ret;
    }
  },

  /*
  model samples, prediction sample
  */
  getBBoxCoordinate(state, getters) {
    return function(class_label, box) {
      let w = box[2]*100
      let h = box[3]*100
      let x = box[0]*100 - w/2
      let y = box[1]*100 - h/2
      x = Math.min(Math.max(x, 0), 100)
      y = Math.min(Math.max(y, 0), 100)

      if (x + w > 100) w = 100 - x;
      if (y + h > 100) h = 100 - y;

      return [class_label, x, y, w, h];
    }
  },
  getLastValidationResults(state, getters) {
    let model = undefined;
    for(let m of state.models) {
      if(m.model_id == state.selected_model_id) {
        model = m;
      }
    }
    if(!model) return;

    const result = model.best_epoch_validation_result;
    if(!result.bbox_path_list) return;

    const path = result.bbox_path_list;
    const label_list = result.bbox_list;
    let ret = []
    for(let i=0; i<path.length; i++){
      let bboxes = []
      if(label_list && label_list.length > 0){
        for(let j=0; j<label_list[i].length; j++) {
          const class_label = label_list[i][j].class;
          const box = label_list[i][j].box;
          bboxes.push(getters.getBBoxCoordinate(class_label, box));
        }
      }
      ret.push({
        "path": path[i],
        "predicted_bboxes": bboxes,
      });
    }
    return ret;
  },
  getPredictResults(state, getters) {
    let result = state.predict_results
    let image_path = result.bbox_path_list
    let label_list = result.bbox_list
    let ret = []

    const i_start = state.predict_page * state.predict_page_image_count;
    let i_end = i_start + state.predict_page_image_count;
    if(i_end > image_path.length) {
      i_end = image_path.length;
    }

    for(let i=i_start; i < i_end; i++) {
      let bboxes = []
      if(label_list && label_list.length > 0) {
        for(let j=0; j < label_list[i].length; j++){
          let class_label = label_list[i][j].class
          let box = label_list[i][j].box
          bboxes.push(getters.getBBoxCoordinate(class_label, box));
        }
      }
      ret.push({
          "path": image_path[i],
          "predicted_bboxes":bboxes
      })
    }
    return ret
  },
  getPageMax(state) {
    if(state.predict_results) {
      return Math.floor(state.predict_results.bbox_path_list.length / state.predict_page_image_count);
    }
  },
}