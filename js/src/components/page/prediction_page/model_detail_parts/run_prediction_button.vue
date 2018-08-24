<template>
  <div id="run-prediction-button">
    <div class="prediction-button" :disabled="!predictModelId" @click="runPrediction">
      <span class="icon">
        <i class="fa fa-arrow-right" aria-hidden="true"></i> Prediction Start
      </span>
      Start!
    </div>
  </div>
</template>

<script>
export default {
  name: 'RunPredictionButton',
  computed: {
    predictModelId: function () {
      if (this.$store.state.project) {
        return this.$store.state.project.deploy_model_id
      }
    }
  },
  methods: {
    runPrediction: function () {
      this.$store.dispatch('runPrediction')
      this.$store.dispatch('updatePredictionInfo')
    }
  }
}
</script>

<style lang="scss" scoped>
#run-prediction-button {
  .prediction-button{
      line-height: $panel-height;
  }
  // $button-height: 64px;
  // $button-bg-color: #7F9DB5;
  //
  // position: absolute;
  // top: 0;
  // width: 100%;
  //
  // .prediction-button {
  //   width: 100%;
  //   height: $button-height;
  //   margin: 0;
  //   // margin-left: 4%;
  //
  //   background-color: $button-bg-color;
  //   color: #ffffff;
  //   text-align: center;
  //   line-height: $button-height;
  //   border-radius: 4px;
  //   cursor: pointer;
  //
  //   .icon {
  //     margin-right: 4px;
  //   }
  // }
}
</style>
