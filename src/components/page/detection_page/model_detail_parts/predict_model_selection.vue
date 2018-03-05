<template>
  <div id="predict-model-selection">
    <div class="button-area">
      <div class="set-predict-model" @click="setPredictModel">
        Deploy Model
      </div>

      <div class="button-description" @mouseenter="hoverDescription(true)" @mouseleave="hoverDescription(false)">
        <i class="fa fa-info-circle" aria-hidden="true"></i>
      </div>

      <div v-if="show_description">
        <div class="balloon">
          If you push this buton,<br>
          you can use current selected<br>
          model in prediction page.
        </div>
        <div class="triangle"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "PredictModelSelection",
  props: {
    "model": {
      type: Object,
      required: true,
    }
  },
  data: function() {
    return {
      "show_description": false,
    }
  },
  computed: {
    stateRunning() {
      return this.$store.state.const.state_id["running"];
    }
  },
  methods: {
    setPredictModel: function() {
      if(this.model.state == this.stateRunning) {
        alert("You can't deploy running model. Please wait or terminate model training.");
        return;
      }

      this.$store.dispatch('deployModel', {"model_id": this.model.model_id});
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
    bottom: 28px;
    right: calc(#{$balloon-border-width});
    border-top: $balloon-border-width solid $balloon-color;
    border-left: $balloon-border-width solid transparent;
    border-right: $balloon-border-width solid transparent;
  }
}
</style>
