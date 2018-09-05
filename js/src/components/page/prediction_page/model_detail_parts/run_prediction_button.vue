<template>
  <div id="run-prediction-button">
    <div class="prediction-button" :disabled="!predictModelId" @click="runPrediction">
      
          <!-- <i class="fa fa-arrow-right" aria-hidden="true"></i> Prediction Start -->
      <img class="title" :src="right_arrow">
    </div>
  </div>
</template>

<script>
export default {
  name: 'RunPredictionButton',
  data: function () {
    return {
      right_arrow: require('../../../../../static/img/PS.svg')
    }
  },
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
    font-family:$content-top-header-font-family;
    font-size:$content-top-header-font-size;
  }

  div {
      align-self: center;
      padding-right: 10px;
      span {
        padding-left: 5px;
        
      }
  }
  .title{
    margin-top: calc(#{$content-top-header-font-size} * 0.3);
    height: calc(#{$content-top-header-font-size} * 1.2 + 5px);
  }


 }
</style>
