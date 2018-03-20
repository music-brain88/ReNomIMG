<template>
  <div id="predict-model-selection">
    <div class="button-area">
      <button class="set-predict-model" v-if="!isPredict" :disabled="model.train_loss_list.length == 0" @click="setPredictModel">
        Deploy Model
      </button>

      <button class="set-predict-model" v-if="isPredict" @click="show_undeploy_dialog=true">
        Undeploy
      </button>

      <div class="button-description" @mouseenter="hoverDescription(true)" @mouseleave="hoverDescription(false)">
        <i class="fa fa-info-circle" aria-hidden="true"></i>
      </div>

      <div v-if="show_description">
        <div class="balloon">
          If you push this button,<br>
          you can use current selected<br>
          model in prediction.
        </div>
        <div class="triangle"></div>
      </div>
    </div>

    <modal-box v-if='show_undeploy_dialog'
      @ok='resetPredictModel'
      @cancel='show_undeploy_dialog=false'>
      <div slot='contents'>
        Would you like to undeploy this model?"
      </div>
      <span slot="okbutton">
        <button id="delete_labels_button" class="modal-default-button"
          @click="resetPredictModel">
          Undeploy
        </button>
      </span>
    </modal-box>

  </div>
</template>

<script>
import * as constant from '@/constant'
import ModalBox from '@/components/common/modalbox'

export default {
  name: "PredictModelSelection",
  components: {
    "modal-box": ModalBox,
  },
  props: {
    "model": {
      type: Object,
      required: true,
    }
  },
  data: function() {
    return {
      show_undeploy_dialog: false,
      show_description: false,
    }
  },
  computed: {
    isPredict() {
      return this.model.model_id == this.$store.state.project.deploy_model_id;
    },
  },
  methods: {
    setPredictModel: function() {
      if(this.model.state == constant.STATE_ID["Running"]) {
        alert("You can't deploy running model. Please wait or terminate model training.");
        return;
      }

      this.$store.dispatch('deployModel', {"model_id": this.model.model_id});
    },
    resetPredictModel: function() {
      this.$store.dispatch('undeployModel', {'model_id': this.model.model_id});
      this.show_undeploy_dialog = false;
    },
    hoverDescription: function(val) {
      this.show_description = val;
    }
  }
}
</script>

<style lang="scss" scoped>
#predict-model-selection {
  $button-bg-color: #7F9DB5;
  $button-bg-color-hover: #7590A5;

  $balloon-top: 84px;
  $balloon-color: #000000;
  $balloon-border-width: 8px;
  $content-margin: 8px;

  .button-area {
    display: flex;
    position: relative;
  }

  .set-predict-model {
    margin-right: $content-margin;
    padding: 4px 16px;
    background-color: $button-bg-color;
    color: #ffffff;
    text-align: center;
    border-radius: 4px;
    cursor: pointer;
  }
  .set-predict-model:hover {
    background-color: $button-bg-color-hover;
  }

  .button-description {
    margin-right: $content-margin;
    line-height: 32px;
  }

  .balloon {
    position: absolute;
    width: 240px;
    top: -$balloon-top;
    right: calc(-#{$balloon-border-width});
    padding: 4px;
    background-color: $balloon-color;
    color: #fff;
    border-radius: 4px;
  }

  .triangle {
    position: absolute;
    top: -4px;
    right: calc(#{$balloon-border-width});
    border-top: $balloon-border-width solid $balloon-color;
    border-left: $balloon-border-width solid transparent;
    border-right: $balloon-border-width solid transparent;
  }
}
</style>
