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
        <p class="value">{{ getAlgorithmName(model.algorithm) }}</p>
      </div>
    </div>

    <div class="model-values">
      <div class="model-iou-map">
        <div class="label-value">
          <p class="label">IoU</p>
          <p class="value value-bold">{{ round_percent(model.best_epoch_iou) }}%</p>
        </div>

        <div class="label-value">
          <p class="label">mAP</p>
          <p class="value value-bold">{{ round_percent(model.best_epoch_map) }}%</p>
        </div>
      </div>

      <div class="model-validation-loss">
        <p class="label">Validation Loss</p>
        <p class="value value-bold">{{ round(model.validation_loss_list[model.best_epoch], 1000) }}</p>
      </div>
    </div>

    <div v-if="isPredict" class="predict_icon">
    deployed
    </div>

    <div v-if="!isPredict" class="delete-button" @click.stop="show_delete_dialog=true">
      <i class="fa fa-times-circle-o" aria-hidden="true"></i>
    </div>

    <modal-box v-if='show_delete_dialog'
      @ok='deleteModel'
      @cancel='show_delete_dialog=false'>
      <div slot='contents'>
        Would you like to delete Model ID: {{this.model.model_id}}?
      </div>
      <span slot="okbutton">
        <button id="delete_labels_button" class="modal-default-button"
          @click="deleteModel">
          Delete
        </button>
      </span>
    </modal-box>

  </div>
</template>

<script>
import * as utils from '@/utils'
import * as constant from '@/constant'
import ModalBox from '@/components/common/modalbox'

export default {
  name: 'ModelListItem',
  components: {
    'modal-box': ModalBox
  },
  data: function () {
    return {
      show_delete_dialog: false
    }
  },
  props: {
    'model': {
      type: Object,
      require: true
    }
  },
  computed: {
    selected () {
      if (this.$store.state.project) {
        return this.model.model_id === this.$store.state.selected_model_id
      }
    },
    isPredict () {
      return this.model.model_id === this.$store.state.project.deploy_model_id
    }
  },
  methods: {
    selectModel: function () {
      this.$store.commit('setSelectedModel', {'model_id': this.model.model_id})
    },
    deleteModel: function () {
      if (this.isPredict) {
        this.$store.commit('setPredictModelId', {
          'model_id': undefined
        })
      }
      if (this.selected) {
        this.$store.commit('setSelectedModel', {'model_id': undefined})
      }
      this.$store.dispatch('deleteModel', {
        'model_id': this.model.model_id
      })
      this.show_delete_dialog = false
    },
    getColor: function (model_state, algorithm) {
      if (model_state === constant.STATE_ID['Reserved']) {
        return constant.STATE_COLOR[model_state]
      } else if (model_state === constant.STATE_ID['Running']) {
        return constant.STATE_COLOR[model_state]
      } else {
        return constant.ALGORITHM_COLOR[algorithm]
      }
    },
    getAlgorithmName: function (id) {
      return constant.ALGORITHM_NAME[id]
    },
    round: function (v, round_off) {
      if (v == null) {
        return '-'
      } else {
        return utils.round(v, round_off)
      }
    },
    round_percent: function (v) {
      if (v == null) {
        return '-'
      } else {
        return utils.round_percent(v)
      }
    }
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
    right: 4px;
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
