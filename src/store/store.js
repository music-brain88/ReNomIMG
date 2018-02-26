import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import Project from './classes/project.js'
import Model from './classes/model.js'
import Epoch from './classes/epoch.js'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    // page name
    page_name: "",

    // projectのクラス
    project: undefined,

    // 予測に使うモデルのID
    predict_model_id: undefined,

    // ナビゲーションバーの表示/非表示フラグ
    navigation_bar_shown_flag: false,

    // モデル作成モーダルの表示/非表示フラグ
    add_model_modal_show_flag: false,

    // show image modal flag
    image_modal_show_flag: false,
    // image data on modal
    image_index_on_modal: undefined,

    class_names: {},

    // predicted result
    predict_results: {"bbox_list": [], "bbox_path_list": []},
    predict_running_flag: false,

    // predict page
    predict_page: 0,
    predict_page_image_count: 20,

    // prediction sample of detection page
    validation_page: 0,

    // csv file name
    csv: "",

    // constant params
    const: {
      algorithm_id: {
        "YOLO": 0,
        "SSD": 1,
      },
      algorithm_name: {
        0: "YOLO",
        1: "SSD",
      },
      algorithm_color: {
        0: "#000068", // Yolo
        1: "#3a8ca9", // SSD
      },
      state_id: {
        "created": 0,
        "running": 1,
        "finished": 2,
        "deleted": 3,
        "validating": 4,
      },
      state_color: {
        0: "",
        1: "#953136",
        2: "#e4b520",
      }
    }
  },

  getters: {
    // get const value
    getStateIdByName(state) {
      return function(name) {
        return state.const.algorithm_id[name];
      }
    },
    getStateIdByName(state) {
      return function(name) {
        return state.const.state_id[name];
      }
    },
    getStateColorByName(state, getters) {
      return function(name) {
        return state.const.state_color[getters.getStateIdByName(name)];
      }
    },
    getAlgorithmIdByName(state) {
      return function(name) {
        return state.const.algorithm_id[name];
      }
    },
    getAlgorithmNameById(state) {
      return function(id) {
        return state.const.algorithm_name[id];
      }
    },
    getAlgorithmColorByName(state, getters) {
      return function(name) {
        return state.const.algorithm_color[getters.getAlgorithmIdByName(name)];
      }
    },
    getColorByStateAndAlgorithm(state, getters) {
      return function(model_state, algorithm) {
        if(model_state == getters.getStateIdByName("running")) {
          return getters.getStateColorByName("running");
        }else{
          return getters.getAlgorithmColorByName(getters.getAlgorithmNameById(algorithm));
        }
      }
    },

    getModels(state, getters) {
      let ret = [];
      const p = state.project;
      if(p) {
        for(let index in p.models) {
          if(p.models[index].state != getters.getStateIdByName("deleted")){
            ret.push(p.models[index]);
          }
        }
      }
      return ret;
    },
    getSelectedModel(state) {
      if(state.project && state.project.selected_model_id) {
        return state.project.getModelFromId(state.project.selected_model_id);
      }
    },
    getRunningModels(state, getters) {
      let ret = [];
      const p = state.project;
      if(p) {
        for(let index in p.models) {
          if(p.models[index].state == getters.getStateIdByName("running") || p.models[index].state == getters.getStateIdByName("validating")){
            ret.push(p.models[index]);
          }
        }
      }
      return ret;
    },
    getFinishedModels(state, getters) {
      let ret = [];
      const p = state.project;
      if(p) {
        for(let index in p.models) {
          if(p.models[index].state == getters.getStateIdByName("finished")){
            ret.push(p.models[index]);
          }
        }
      }
      return ret;
    },
    getModelCounts(state, getters) {
      let ret = {"YOLO": 0, "SSD": 0, "Running": 0};
      const p = state.project;
      if(p) {
        for(let index in p.models) {
          if(p.models[index].state != getters.getStateIdByName("deleted")){
            if(p.models[index].state == getters.getStateIdByName("running")){
              ret["Running"] += 1;
            }else{
              ret[getters.getAlgorithmNameById(p.models[index].algorithm)] += 1;
            }
          }
        }
      }
      return ret;
    },
    getModelLoss(state) {
      if(state.project && state.project.selected_model_id) {
        let s = state.project.getModelFromId(state.project.selected_model_id);
        let loss = {
          "train_loss": s.getTrainLoss(),
          "validation_loss": s.getValidationLoss()
        }
        return loss
      }
    },
    getModelsByStateAndAlgorithm(state, getters) {
      return function(model_state, algorithm) {
        let ret = [];
        const p = state.project;
        if(p) {
          for(let index in p.models) {
            if(p.models[index].state == model_state && p.models[index].algorithm == algorithm){
              ret.push(p.models[index]);
            }
          }
        }
        return ret;
      }
    },
    removeSelectedModelFromModels(state, getters) {
      return function(models) {
        let ret = [];
        const p = state.project;
        if(p) {
          for(let index in models) {
            if(models[index].model_id != p.selected_model_id){
              ret.push(models[index]);
            }
          }
        }
        return ret;
      }
    },
    getPlotData(state, getters){
      return function(models, color, style="circle", size=4) {
        let datas = {
          data:[],
          backgroundColor: "",
          backgroundColor: color,
          pointRadius: size,
          pointHoverRadius: size+2,
          pointStyle: style,
        };

        for(let index in models) {
          let model = models[index];
          if(model.epochs && model.epochs.length > 0 && model.best_epoch != undefined) {
            let model_coordinate = {
              "model_id": model.model_id,
              "algorithm": model.algorithm,
              "algorithm_name": getters.getAlgorithmNameById(model.algorithm),
              "iou_value": model.getRoundedIoU()+"%",
              "map_value": model.getRoundedMAP()+"%",
              "x": model.epochs[model.best_epoch].iou_value*100,
              "y": model.epochs[model.best_epoch].map_value*100,
            }
            datas.data.push(model_coordinate);
          }
        }
        return datas;
      }
    },
    getPlotDataset(state, getters) {
      let dataset = {
        "dataset": [],
        "colors": [],
      };

      // add running model plot data
      let state_name = "running";
      let models = getters.removeSelectedModelFromModels(getters.getRunningModels);
      let color = getters.getStateColorByName(state_name);
      let plotData = getters.getPlotData(models, color);
      dataset.dataset.push(plotData);
      dataset.colors.push(color);

      // add finished model plot data
      // TODO: loop by algorithm
      state_name = "finished";
      let alg_name = "YOLO";
      let state_id = getters.getStateIdByName(state_name);
      let alg_id = getters.getAlgorithmIdByName(alg_name);
      models = getters.removeSelectedModelFromModels(getters.getModelsByStateAndAlgorithm(state_id, alg_id));
      color = getters.getAlgorithmColorByName(alg_name);
      plotData = getters.getPlotData(models, color);
      dataset.dataset.push(plotData);
      dataset.colors.push(color);

      // add selected model plot data
      let model = getters.getSelectedModel;
      if(model) {
        const selected_color = "#999999";
        plotData = getters.getPlotData([model], selected_color, "rectRot", 10);
        dataset.dataset.push(plotData);
        dataset.colors.push(color);
      }

      return dataset;
    },
    getSelectedModelId(state) {
      if(state.project) {
        return state.project.selected_model_id;
      }
    },
    getNavigationBarShowFlag(state) {
      return state.navigation_bar_shown_flag;
    },
    getLabelDict(state) {
      return state.class_names
    },
    getPredictModel(state) {
      if(state.project) {
        return state.project.getModelFromId(state.predict_model_id);
      }
    },
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
      let model = getters.getSelectedModel
      if (!model)
        return

      let epochs = model.epochs
      if(epochs && epochs.length > 0){
        let result
        for (let i=1 ; i <= epochs.length; i++) {
          result = epochs[epochs.length - i].validation_result
	  if(result.bbox_path_list && result.bbox_list && result.bbox_list.length){
            break
	  }
	}
	if (!result.bbox_path_list)return
        // Pickup some images here.
        // This method returns true bboxes and base64 image data.
        let path = result.bbox_path_list
        let label_list = result.bbox_list
        let ret = []
        for(let i=0; i < path.length; i++) {
          let bboxes = []
          if(label_list && label_list.length > 0) {
            for(let j=0; j < label_list[i].length; j++){
              let class_label = label_list[i][j].class
              let box = label_list[i][j].box
              bboxes.push(getters.getBBoxCoordinate(class_label, box));
            }
          }
          ret.push({
              "path": path[i],
              "predicted_bboxes":bboxes
          })
        }
        return ret
      }
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
  },

  mutations: {
    setPageName(state, payload) {
      state.page_name = payload.page_name;
    },
    setProject(state, payload) {
      if(!state.project || state.project.project_id != payload.project_id) {
        const project = new Project(payload.project_id, payload.project_name, payload.project_comment);
        state.project = project;
      }
    },
    setModels(state, payload) {
      if(!state.project.models || state.project.models.length == 0){
        state.project.createModels(payload.models);
      }else{
        state.project.updateModels(payload.models);
      }
    },
    setEpochs(state, payload) {
      let s = state.project.getModelFromId(payload.model_id);
      if(!s.epochs || Object.keys(s.epochs).length < Object.keys(payload.epochs).length){
        s.epochs = [];
        for(let i in payload.epochs) {
          let d = payload.epochs[i];

          let validation_results = {};
          if(i == s.best_epoch || i == Object.keys(payload.epochs).length - 1) {
            validation_results = d.validation_results
          }

          s.epochs.push(new Epoch(d.epoch_id, d.nth_epoch, d.train_loss, d.validation_loss, d.weight, d.iou_value, d.map_value, validation_results));
        }
      }
    },
    setSelectedModel(state, payload) {
      state.project.selected_model_id = payload.model_id;
    },
    setNavigationBarShowFlag(state, payload) {
      state.navigation_bar_shown_flag = payload.flag;
    },
    sortModels(state, payload) {
      state.project.sortModels(payload.sort_by);
    },
    setAddModelModalShowFlag(state, payload) {
      state.add_model_modal_show_flag = payload.add_model_modal_show_flag;
    },
    stopModel(state, payload) {
      let m = state.project.getModelFromId(payload.model_id);
      m.state = 2;
    },
    setModelStopFlag(state, payload) {
      // Dirty structure.
      let m = state.project.getModelFromId(payload.model_id);
      m.state = 4;
    },
    setPredictModelId(state, payload) {
      if(state.project){
        let m = state.project.getModelFromId(payload.model_id);
        if(m.state == state.const.state_id["running"]) {
          state.predict_model_id  =undefined;
        }else{
          state.predict_model_id = payload.model_id;
        }
      }
    },
    setDatasetInfov0(state, payload) {
      state.class_names = payload.class_names
    },
    setCurrentLearningInfo(state,payload) {
      let model = state.project.getModelFromId(payload.model_id);
      model.current_learning_info = payload.learning_info;
    },
    setImageModalShowFlag(state, payload) {
      state.image_modal_show_flag = payload.flag;
    },
    setImageIndexOnModal(state, payload) {
      state.image_index_on_modal = payload.index;
    },
    setPredictResult(state, payload) {
      state.predict_running_flag = false
      state.predict_results = payload.predict_results;
      state.csv = payload.csv;
    },
    setPredictPage(state, payload) {
      const max_chunk = Math.floor(state.predict_results.bbox_path_list.length / state.predict_page_image_count);
      if(payload.page > max_chunk) {
        state.predict_page = max_chunk;
      }else if(payload.page < 0) {
        state.predict_page = 0;
      }else{
        state.predict_page = payload.page;
      }
    },
    setValidationPage(state, payload) {
      state.validation_page = payload.page
    },
    setPredictRunningFlag(state, payload) {
      state.predict_running_flag = payload.flag
    },
    resetPredictResult(state, payload) {
      state.predict_results = {"bbox_list": [], "bbox_path_list": []};
    },
  },

  actions: {
    async loadProject(context, payload) {
      let url = "/api/obj_detector/v1/projects/" + payload.project_id
      return axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }

          context.commit("setProject", {
            "project_id": response.data.project_id,
            "project_name": response.data.project_name,
            "project_comment": response.data.project_comment,
          });
        });
    },
    async loadProjectModels(context, payload) {
      await context.dispatch("loadProject", {
        "project_id": payload.project_id,
      });
      await context.dispatch("loadDatasetInfov0");

      let url = "/api/obj_detector/v1/projects/" + payload.project_id + "/models"
      return axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }

          context.commit("setModels", {
            "models": response.data,
          });
        });
    },
    async loadDatasetInfov0(context, payload){
      let url = "/api/obj_detector/v1/dataset_info"
      return await axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }

          context.commit("setDatasetInfov0", {
            "class_names": response.data['class_names']
          })
        });
    },
    async loadProjectData(context, payload) {
      await context.dispatch("loadProjectModels", {
        "project_id": payload.project_id,
      });
      for(let index in context.state.project.models){
        let url = "/api/obj_detector/v1/projects/" + payload.project_id + "/models/" + context.state.project.models[index].model_id + "/epochs"
        axios.get(url)
          .then(function(response) {
            if(response.data.error_msg) {
              alert("Error: " + response.data.error_msg);
              return;
            }

            context.commit("setEpochs", {
              "model_id": context.state.project.models[index].model_id,
              "epochs": response.data.epochs,
            })
            context.commit("setCurrentLearningInfo", {
              "model_id": context.state.project.models[index].model_id,
              "project_id": context.state.project.project_id,
              "learning_info": response.data.running_info,
            })
          });
      }
      if(localStorage.getItem("predictModelId") && parseInt(localStorage.getItem("predictModelId")) != context.state.predict_model_id) {
        context.commit("setPredictModelId",{
          "model_id": parseInt(localStorage.getItem("predictModelId")),
        });
      }
    },
    async loadOriginalImage(context, payload) {
      let url = "/api/obj_detector/v1/original_img"
      let fd = new FormData()
      fd.append('root_dir', payload.img_path)
      return await axios.post(url, fd)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }
        });
    },
    deleteModel(context, payload) {
      context.state.project.removeModelFromId(payload.model_id);

      let url = "/api/obj_detector/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id
      return axios.delete(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }
        });
    },
    async createModel(context, payload) {
      // add fd model data
      let fd = new FormData();
      fd.append('total_epoch', payload.total_epoch);
      fd.append('seed', payload.seed);
      fd.append('algorithm', payload.algorithm);
      fd.append('hyper_parameter', payload.hyper_parameter);

      let url = "/api/obj_detector/v1/projects/" + context.state.project.project_id + "/models"
      return axios.post(url, fd);
    },
    async runModel(context, payload) {
      let hyper_parameter = JSON.stringify({
        "image_width": payload.image_width,
        "image_height": payload.image_height,
        "batch_size": payload.batch_size,
        "additional_params": payload.additional_params,
      })
      const result = await context.dispatch("createModel", {
        'algorithm': payload.algorithm,
        'total_epoch': payload.total_epoch,
        'seed': payload.seed,
        'hyper_parameter': hyper_parameter,
      });
      if(result.data.error_msg) {
        alert(result.data.error_msg);
        return;
      }

      const model_id = result.data.model_id;
      context.state.project.models.unshift(new Model(model_id, context.state.project.project_id, 1, payload.total_epoch, payload.seed, payload.algorithm, hyper_parameter, 1, undefined));
      context.commit("setModels", context.state.project.models);

      const url = "/api/obj_detector/v1/projects/" + context.state.project.project_id + "/models/" + model_id + "/run";
      axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }
        });
    },
    stopModel(context, payload) {
      const url = "/api/obj_detector/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id + "/stop";
      axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            alert("Error: " + response.data.error_msg);
            return;
          }

          context.commit("stopModel", {
              "model_id": payload.model_id,
            })
        });
    },
    runPrediction(context, payload) {
      if(context.state.project) {
        context.commit('setPredictRunningFlag', {'flag': true})
        const url = "/api/obj_detector/v1/projects/" + context.state.project.project_id + "/models/" + context.state.predict_model_id + "/run_prediction";
        axios.get(url)
          .then(function(response) {
            if(response.data.error_msg) {
              alert("Error: " + response.data.error_msg);
              return;
            }

            context.commit("setPredictResult", {
              "predict_results": response.data.predict_results,
              "csv": response.data.csv,
            })
          });
      }
    }
  }
})

export default store
