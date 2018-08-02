<template>
  <div id="predict-model-selection">
    <div class="button-area">
      <div class="set-predict-model" v-if="!isPredict" :disabled="model.state !== 2" @click="setPredictModel">
        <i class="fa fa-angle-right icon-navgation"></i>Deploy Model <span class="tooltip" data-html="true" aria-label="If you push this button, you can use current selected model in prediction."><i class="fa fa-info button-description"></i></span>
      </div>

      <div class="set-predict-model" v-if="isPredict" @click="show_undeploy_dialog=true">
        <i class="fa fa-angle-right icon-navgation"></i>Undeploy
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
      <div class="modal-contents" slot='contents'>
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
  name: 'PredictModelSelection',
  components: {
    'modal-box': ModalBox
  },
  props: {
    'model': {
      type: Object,
      required: true
    }
  },
  data: function () {
    return {
      show_undeploy_dialog: false,
      show_description: false
    }
  },
  computed: {
    isPredict () {
      return this.model.model_id === this.$store.state.project.deploy_model_id
    }
  },
  methods: {
    setPredictModel: function () {
      if (this.model.state === constant.STATE_ID['Running']) {
        alert("You can't deploy running model. Please wait or terminate model training.")
        return
      }

      this.$store.dispatch('deployModel', {'model_id': this.model.model_id})
    },
    resetPredictModel: function () {
      this.$store.dispatch('undeployModel', {'model_id': this.model.model_id})
      this.show_undeploy_dialog = false
    },
    hoverDescription: function (val) {
      this.show_description = val
    }
  }
}
</script>

<style lang="scss" scoped>
#predict-model-selection {
  $balloon-top: 84px;
  $balloon-color: #000000;
  $balloon-border-width: 8px;

  .button-area {
    margin-left:$content-parts-margin;
    height: $content-top-header-hight;
  }

  .set-predict-model {
    display: flex;
    line-height: $content-top-header-hight;
    color: #ffffff;
    text-align: center;
    cursor: pointer;
  }

  .icon-navgation{
    margin-right: $content-parts-margin;
    line-height: $content-top-header-hight;
    // margin-top: calc(#{$content-top-header-hight}*0.25);
  }

  .button-description {
    margin-right: $content-parts-margin;
    line-height: $content-top-header-hight;
    border-color: $font-color;
    // margin-top: calc(#{$content-top-header-hight}*0.25);
  }

  .balloon {
    position: relative;
    width: 240px;
    top: -$balloon-top*2;
    right: calc(-#{$balloon-border-width});
    padding: 4px;
    background-color: $balloon-color;
    color: #fff;
    border-radius: 4px;

    &:after{
      content: "";
      position: relative;
      right: 0;
      bottom: -20px;
      width: 0px;
      height: 0px;
      margin: auto;
      border-style: solid;
      border-color: $balloon-color transparent transparent transparent;
      border-width: 20px 20px 0 20px;
    }
  }

  .triangle {
    position: absolute;
    top: -4px;
    right: calc(#{$balloon-border-width});
    border-top: $balloon-border-width solid $balloon-color;
    border-left: $balloon-border-width solid transparent;
    border-right: $balloon-border-width solid transparent;
  }
  .modal-contents{
    color:#000000;
  }
  span{
    &:after{
      width:147px;
      white-space: pre-line;
    }
  }
}
</style>
