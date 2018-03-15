import axios from 'axios'
import * as constant from '@/constant'

export default {
  /*
  initialize data
  */
  async initData(context, payload){
    await context.dispatch("loadProject", {"project_id": payload.project_id});
    await context.dispatch("loadModels", {"project_id": payload.project_id});
  },

  async loadProject(context, payload) {
    const url = "/api/renom_img/v1/projects/" + payload.project_id
    return axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }
        context.commit("setProject", {
          "project_id": response.data.project_id,
          "project_name": response.data.project_name,
          "project_comment": response.data.project_comment,
          "deploy_model_id": response.data.deploy_model_id,
        });
      });
  },

  async loadModels(context, payload) {
    await context.dispatch('loadProject',{"project_id": payload.project_id});

    const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models"
    return axios.get(url, {
      timeout: 60000,
      params: {
        "model_count": context.state.models.length,
        "deploy_model_id": context.state.project.deploy_model_id,
      }
    }).then(function(response) {
      if(response.data.error_msg) {
        context.commit('setAlertModalFlag', {'flag': true});
        context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
        return;
      }

      context.commit("setModels", {"models": response.data});
      context.dispatch('loadModels', {"project_id": payload.project_id});
    }).catch(function(error) {
      context.dispatch('loadModels', {"project_id": payload.project_id});
    });
  },

  /*
  model list area
  */
  // check weight exists on server
  async checkWeightExist(context, payload) {
    if(!context.state.weight_exists) {
      context.commit('setWeightDownloadModal', {'weight_downloading_modal': true});
      const url = "/api/renom_img/v1/weights/yolo";
      return axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true});
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
            return;
          }

          if(response.data.weight_exist == 1) {
            context.commit('setWeightExists', {'weight_exists': true});
            context.commit('setWeightDownloadModal', {'weight_downloading_modal': false});
          }
        });
    }
  },

  // check weight downloading process
  async checkWeightDownloadProgress(context, payload) {
    if(!context.state.weight_exists) {
      let url = "/api/renom_img/v1/weights/yolo/progress/"+payload.i;
      return axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true});
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
            return;
          }

          if(response.data.progress) {
            context.commit('setWeightDownloadProgress', {'progress': response.data.progress});
          }
          if(response.data.progress >= 100) {
            context.commit('setWeightExists', {'weight_exists': true});
            context.commit('setWeightDownloadModal', {'weight_downloading_modal': false});
          }
        });
    }
  },

  // create model before run model
  async createModel(context, payload) {
    // add fd model data
    let fd = new FormData();
    fd.append('hyper_parameters', payload.hyper_parameters);
    fd.append('algorithm', payload.algorithm);
    fd.append('algorithm_params', payload.algorithm_params);

    let url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models"
    return axios.post(url, fd);
  },

  // run model
  async runModel(context, payload) {
    await context.dispatch('checkWeightExist');
    for(let i=1; i<=10; i++) {
      await context.dispatch('checkWeightDownloadProgress', {'i': i});
    }

    const hyper_parameters = JSON.stringify(payload.hyper_parameters);
    const algorithm_params = JSON.stringify(payload.algorithm_params);
    const result = await context.dispatch("createModel", {
      'hyper_parameters': hyper_parameters,
      'algorithm': payload.algorithm,
      'algorithm_params': algorithm_params,
    });
    if(result.data.error_msg) {
      context.commit('setAlertModalFlag', {'flag': true});
      context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
      return;
    }

    const model_id = result.data.model_id;

    const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + model_id + "/run";
    axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }
      });
  },

  // delete model
  deleteModel(context, payload) {
    let url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id
    return axios.delete(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }
      });
  },

  /*
  model progress
  */
  stopModel(context, payload) {
    const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id + "/stop";
    axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }
      });
  },

  /*
  model detail
  */
  deployModel(context, payload) {
    const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id + "/deploy";
    axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }

        context.commit("setDeployModelId", {
          "model_id": payload.model_id,
        });
      });
  },
  undeployModel(context, payload) {
    const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + payload.model_id + "/undeploy";
    axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }

        context.commit("setDeployModelId", {
          "model_id": undefined,
        });
      });
  },

  /*
  tag list
  */
  async loadDatasetInfov0(context, payload){
    let url = "/api/renom_img/v1/dataset_info"
    return await axios.get(url)
      .then(function(response) {
        if(response.data.error_msg) {
          context.commit('setAlertModalFlag', {'flag': true});
          context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
          return;
        }

        context.commit("setDatasetInfov0", {
          "class_names": response.data['class_names']
        })
      });
  },


  /*
  prediction
  */
  runPrediction(context, payload) {
    if(context.state.project) {
      context.commit('setPredictRunningFlag', {'flag': true})
      const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + context.state.project.deploy_model_id + "/run_prediction";
      axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true});
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
            context.commit('setPredictRunningFlag', {'flag': false});
            return;
          }

          context.commit("setPredictResult", {
            "predict_results": response.data.predict_results,
            "csv": response.data.csv,
          })
        });
    }
  },
  updatePredictionInfo(context, payload) {
    if(context.state.project) {
      const url = "/api/renom_img/v1/projects/" + context.state.project.project_id + "/models/" + context.state.project.deploy_model_id + "/prediction_info";
      axios.get(url)
        .then(function(response) {
          if(response.data.error_msg) {
            context.commit('setAlertModalFlag', {'flag': true});
            context.commit('setErrorMsg', {'error_msg': response.data.error_msg});
            return;
          }

          context.commit("setPredictInfo", {
            "predict_total_batch": response.data.predict_total_batch,
            "predict_last_batch": response.data.predict_last_batch,
          });
          if(context.state.predict_running_flag) {
            context.dispatch('updatePredictionInfo');
          }
        });
    }
  },
}