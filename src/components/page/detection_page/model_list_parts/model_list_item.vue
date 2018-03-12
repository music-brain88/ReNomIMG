<template>
  <div id="model-list-item" :class="{ active: selected }" @click="selectModel">
    <div class="model-state" v-bind:style="{backgroundColor: getColor(model.state, model.algorithm)}"></div>

    <div class="model-id-algorithm">
      <div class="label-value">
        <p class="label">Model ID</p>
        <p class="value">{{ model.model_id }}</p>
      </div>
      <div class="label-value">
        <p class="label">Algorithm</p>
        <p class="value">{{ getAlgorithmNameById(model.algorithm) }}</p>
      </div>
    </div>

    <div class="model-values">
      <div class="model-iou-map">
        <div class="label-value">
          <p class="label">IoU</p>
          <p class="value value-bold">{{ model.getRoundedIoU() }}%</p>
        </div>

        <div class="label-value">
          <p class="label">mAP</p>
          <p class="value value-bold">{{ model.getRoundedMAP() }}%</p>
        </div>
      </div>

      <div class="model-validation-loss">
        <p class="label">Validation Loss</p>
        <p class="value value-bold">{{ model.getRoundedValidationLoss() }}</p>
      </div>
    </div>

    <div v-if="isPredict" class="predict_icon">
    deployed
    </div>

    <div v-if="!isPredict" class="delete-button" @click.stop="deleteModel">
      <i class="fa fa-times-circle-o" aria-hidden="true"></i>
    </div>
  </div>
</template>

<script>
export default {
  name: "ModelListItem",
  props: {
    "model": {
      type: Object,
      require: true
    }
  },
  computed: {
    selected() {
      if(this.$store.state.project) {
        return this.model.model_id == this.$store.state.project.selected_model_id;
      }
    },
    isPredict() {
      return this.model.model_id == this.$store.state.project.deploy_model_id;
    },
  },
  methods: {
    selectModel: function() {
      this.$store.commit("setSelectedModel", {
        "model_id": this.model.model_id
      });
    },
    deleteModel: function() {
      let confirm_text = "Would you like to delete Model ID: "+ this.model.model_id +"?"
      if(confirm(confirm_text)){
        if(this.isPredict) {
          this.$store.commit("setPredictModelId", {
            "model_id": undefined,
          });
        }
        if(this.selected) {
          this.$store.commit("setSelectedModel", {"model_id": undefined});
        }
        this.$store.dispatch("deleteModel", {
          "model_id": this.model.model_id
        });
      }
    },
    getColor: function(model_state, algorithm) {
      return this.$store.getters.getColorByStateAndAlgorithm(model_state, algorithm);
    },
    getAlgorithmNameById: function(id) {
      return this.$store.getters.getAlgorithmNameById(id);
    },
  }
}
</script>

<style lang="scss" scoped>
#model-list-item {
  $item-margin-bottom: 8px;
  $state-width: 8px;
  $label-color: #999999;
  $label-color-hover: #666666;
  $label-size: 12px;

  display: flex;
  position: relative;
  width: calc(100% - 16px);
  height: 95px;
  margin: 0px 0px $item-margin-bottom;
  box-shadow: 1px 1px #ddd;
  background-color: #ffffff;

  .predict_icon {
    position:absolute;
    bottom: 0;
    right: 0;
    padding: 0 4px;
    background-color: #c13d18;
    color: #fff;
    font-size: 12px;
  }

  .label {
    margin: 0;
    font-size: $label-size;
    color: $label-color;
  }
  .value {
    margin: 0;
  }
  .value-bold {
    font-weight: bold;
  }

  .model-state {
    flex-grow: 0;
    width: $state-width;
    height: 100%;
  }

  .label-value {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    padding: 2px 8px;
  }

  .model-id-algorithm {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
  }

  .model-values {
    flex-grow: 1;
    display: flex;
    flex-direction: column;

    .model-iou-map {
      flex-grow: 1;
      display: flex;
    }
    .model-validation-loss {
      flex-grow: 1;
      padding: 2px 8px;
    }
  }

  .delete-button {
    position: absolute;
    bottom: 0;
    right: 0;
    color: $label-color;
  }
  .delete-button:hover {
    color: $label-color-hover;
  }
}

#model-list-item:hover {
  background-color: #eeeeee;
}
#model-list-item.active {
  background-color: #d1d1d1;
}
</style>

