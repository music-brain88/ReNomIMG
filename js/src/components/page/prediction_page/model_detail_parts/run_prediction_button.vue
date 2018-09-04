<template>
  <div id="run-prediction-button">
    <div class="prediction-button" :disabled="!predictModelId" @click="runPrediction">
      
      <div>
        <span>
          <!-- <i class="fa fa-arrow-right" aria-hidden="true"></i> Prediction Start -->
          <img class="right_arrow" :src="right_arrow">Prediction Start
        </span>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: 'RunPredictionButton',
  data: function () {
    return {
      right_arrow: require('../../../../../static/img/yajiright.png')
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
  }

  div {
      align-self: center;
      padding-right: 10px;
      span {
        padding-left: 5px;
        font-size: 1.4rem;
      }
  }

 }
</style>
