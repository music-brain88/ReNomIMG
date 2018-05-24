<template>
  <div id="model-progress" v-bind:class='{emphasizeItem: model.model_id==$store.state.selected_model_id}'>
    <div class="value-item">
      <div class="label">
        Model ID
      </div>
      <div class="value">
        {{ model.model_id }}
      </div>
    </div>

    <div class="value-item">
      <div class="label">
        Epoch
      </div>
      <div class="value"
        v-if="model.running_state!==running_state['starting']">
        {{model.validation_loss_list.length}}/{{model.hyper_parameters['total_epoch']}}
      </div>
      <div class="value"
        v-if="model.running_state===running_state['starting']">
        -/-
      </div>
    </div>

    <div class="value-item">
      <div class="label">
        Batch
      </div>
      <div class="value"
        v-if="model.running_state!==running_state['starting']">
        {{model.last_batch}}/{{model.total_batch}}
      </div>
      <div class="value"
        v-if="model.running_state===running_state['starting']">
        -/-
      </div>
    </div>

    <div class="progress-bar-area">
      <div class="progress-bar-back">
        <div class="progress-bar">
          <div class="progress-bar-animation"
            v-bind:style="{backgroundColor: progress_bar_color}"
            v-bind:class="{animated: model.running_state===running_state['training']}">
          </div>
        </div>
      </div>
      <div class="progress-bar-mask" v-if="model.running_state===running_state['training']"></div>
    </div>

    <div class="value-item">
      <div class="label">
        <span v-if="model.running_state===running_state['starting']">
          Starting...
        </span>
        <span v-if="model.running_state===running_state['training']">
          Train Loss
        </span>
        <span v-if="model.running_state===running_state['validating']">
          Validating
        </span>
        <span v-if="model.running_state===running_state['stopping']">
          Stopping...
        </span>
      </div>

      <div class="value">
        <span v-if="model.running_state===running_state['training']">
          {{round(model.last_train_loss, 1000).toFixed(3)}}
        </span>
        <span v-if="model.running_state!==running_state['training']">
          <i class="fa fa-spinner fa-spin" aria-hidden="true"></i>
        </span>
      </div>
    </div>

    <div class="stop-button-area">
      <div class="stop-button" @click="show_stop_dialog=true"
        v-if="model.running_state!==running_state['stopping']">
        <i class="fa fa-pause-circle-o" aria-hidden="true"></i>
      </div>
    </div>

    <modal-box v-if='show_stop_dialog'
      @ok='stopModel'
      @cancel='show_stop_dialog=false'>
      <div slot='contents'>
        Would you like to stop Model ID: {{this.model.model_id}}?
      </div>
      <span slot="okbutton">
        <button id="delete_labels_button" class="modal-default-button"
          @click="stopModel">
          Stop
        </button>
      </span>
    </modal-box>
  </div>
</template>

<script>
import * as utils from '@/utils'
import ModalBox from '@/components/common/modalbox'

export default {
  name: 'modelProgress',
  components: {
    'modal-box': ModalBox
  },
  data: function () {
    return {
      progress_bar_color: '#953136',
      running_state: {
        'training': 0,
        'validating': 1,
        'starting': 3,
        'stopping': 4
      },
      show_stop_dialog: false
    }
  },
  props: {
    'index': {
      type: Number,
      require: true
    },
    'model': {
      type: Object,
      require: true
    }
  },
  mounted: function () {
    this.updateProgressBar()
  },
  updated: function () {
    this.updateProgressBar()
  },
  methods: {
    updateProgressBar: function () {
      const progress_bar_elm = document.getElementsByClassName('progress-bar-back')
      if (progress_bar_elm.length === 0) return

      const progress_bar_width = progress_bar_elm[0].clientWidth
      let current_width = this.model.last_batch / this.model.total_batch * progress_bar_width
      let e = document.getElementsByClassName('progress-bar')
      if (e && e[this.index]) {
        e[this.index].style.width = current_width + 'px'
      }
    },
    stopModel: function () {
      this.$store.dispatch('stopModel', {
        'model_id': this.model.model_id
      })
      this.model.running_state = this.running_state['stopping']
      this.show_stop_dialog = false
    },
    round: function (v, round_off) {
      return utils.round(v, round_off)
    }
  }
}
</script>

<style lang="scss" scoped>
#model-progress {
  $content-margin: 8px;

  $label-color: #666666;
  $label-font-size: 12px;
  $value-font-size: 14px;

  $progress-bar-width: 24%;
  $progress-bar-height: 12px;
  $progress-bar-bg-color: #e8e8e8;
  $progress-bar-color: #000099;

  $stop-button-color: #999999;
  $stop-button-color-hover: #666666;

  width: 100%;
  display: flex;
  display: -webkit-flex;

  margin-top: $content-margin;

  .value-item {
    display: flex;
    display: -webkit-flex;
    flex-direction: column;
    -webkit-flex-direction: column;

    width: 64px;

    .label, .label span {
      color: $label-color;
      font-size: $label-font-size;
      line-height: $label-font-size;
    }

    .value, .value span {
      font-size: $value-font-size;
      line-height: $value-font-size;
    }
  }

  .stop-button-area {
    position: relative;
    .stop-button {
      position: absolute;
      bottom: 0;

      margin-right: $content-margin;
      line-height: $value-font-size;
      font-size: $value-font-size;
      color: $stop-button-color;
    }
    .stop-button:hover {
      color: $stop-button-color-hover;
    }
  }

  .progress-bar-area {
    position: relative;
    width: $progress-bar-width;
    height: calc(#{$label-font-size} + #{$value-font-size});
    margin-right: $content-margin;

    .progress-bar-back {
      position: absolute;
      bottom: 2px;
      width: 100%;
      height: $progress-bar-height;
      background-color: $progress-bar-bg-color;
      border-radius: 1px;
      border-width: 1px;
      border-color: #a5a5a5;
      border-style: solid;
    }

    .progress-bar {
      position: absolute;
      height: 10px;
      .progress-bar-animation {
        border-radius: 1px;
        background-color: $progress-bar-color;
        height: 100%;
      }
    }

    .progress-bar-mask {
      position: absolute;
      bottom: 3px;
      width: 100%;
      height: calc(#{$progress-bar-height} - 2px);
      background: linear-gradient(70deg, rgba(255, 255, 255, 0.0), 30%, rgba(200, 200, 200, 1), 50%, rgba(255, 255, 255, 0));
      background-size: 50% 100%;
      background-repeat: no-repeat;
      animation: movegrad 2s infinite linear;
      @keyframes movegrad {
          0%{background-position: -100% 0;}
          100%{background-position: 200% 0;}
      }
    }
  }
}
.emphasizeItem {
    animation: emphasize 0.4s;
    animation-iteration-count: 1;
    animation-delay: 0.05s;
}
@keyframes emphasize {
    0%{background-color: #ffffff;}
    30%{background-color: rgba(0, 0, 0, 0.1);}
    100%{background-color: #ffffff;}
}
</style>
