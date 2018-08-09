<template>
  <div id="model-list-item" :class="{ active: selected }" @click="selectModel">
    <div class="model-state" v-bind:style="{backgroundColor: getColor(model.state, model.algorithm)}"></div>

    <div class="model-id-algorithm">
      <div class="row space-top">
        <div class="col-md-12">
          <div class="label-value">
            Model ID&nbsp;&nbsp;<span class="value">{{ model.model_id }}</span>&nbsp;&nbsp;&nbsp;&nbsp;Algorithm&nbsp;&nbsp;<span class="value">{{ getAlgorithmName(model.algorithm) }}</span>
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col-md-6">
          <div class="label-value">
            IoU&nbsp;&nbsp;<span class="value">{{ round_percent(model.best_epoch_iou) }}%</span>
          </div>
        </div>
        <div class="col-md-6">
          mAP&nbsp;&nbsp;<span class="value">{{ round_percent(model.best_epoch_map) }}%</span>
        </div>
      </div>
      <div class="row space-bottom">
        <div class="col-md-12">
          <div class="label-value">
            Validation Loss&nbsp;&nbsp;<span class="value">{{ round(model.validation_loss_list[model.best_epoch], 1000) }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-if="isPredict" class="predict_icon">
    deployed
    </div>

    <div v-if="!isPredict" class="delete-button" @click.stop="show_delete_dialog=true">
      <!-- <i class="fa fa-times-circle-o" aria-hidden="true"></i> -->
      &times;
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
        this.$store.commit('resetPredictResult', {
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
      } else if (model_state === constant.STATE_ID['Created']) {
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
  $state-width: 5px;
  $label-color: #999999;
  $label-color-hover: #CCCCCC;

  display: flex;
  position: relative;
  // width: calc(100% - 16px);
  width: calc(100% - 5px);
  height: 90px;
  margin: 0px 0px $item-margin-bottom;
  background-color: #ffffff;
  color: $label-color;
  font-size: $content-inner-box-font-size;

  .predict_icon {
    position:absolute;
    bottom: 0;
    right: 0;
    padding: 0 4px;
    color: $panel-bg-color;
    font-size: $content-inner-box-font-size;
  }

  .label {
    margin: 0;
    font-size: $content-inner-box-font-size;
    color: $label-color;
  }
  .value {
    color:#000000;
  }

  .model-state {
    flex-grow: 0;
    width: $state-width;
    height: 100%;
  }

  .label-value {
    flex-direction: column;
    padding: 2px 8px;
  }

  .model-id-algorithm {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
  }

  .delete-button {
    position: absolute;
    bottom: 0;
    right: 4px;
    color: $label-color;
    cursor: pointer;
    &:hover{
      color:#FFFFFF;
    }
  }
  .space-top{
    padding-top: 10px;
  }
  .space-bottom{
    padding-bottom: 10px;
  }
}

#model-list-item:hover {
  background-color: #CCCCCC;
}
#model-list-item.active {
  // background-color: #d1d1d1;
  border: solid 1px $panel-bg-color;
}
</style>
