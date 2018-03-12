<template>
  <div id="prediction-result">
    <div class="result-head">
      <div class="title">
        Prediction result
      </div>
      <pager></pager>
    </div>

    <div v-if="getPredictResults.length > 0" class="content">
      <sample-image
        v-for="(item, index) in getPredictResults"
        :key="index"
        :image_path="item.path"
        :bboxes="item.predicted_bboxes"
        :index="index">
      </sample-image>
    </div>

    <div id='loading' v-if='$store.state.predict_running_flag'>
      <div class="loading">
        <div class="animation">
          <div class="bar bar1"></div>
          <div class="bar bar2"></div>
          <div class="bar bar3"></div>
          <div class="bar bar4"></div>
          <div class="bar bar5"></div>
        </div>
      </div>
      <div class="prediction-progress">
        Predicting {{$store.state.predict_last_batch}} / {{$store.state.predict_total_batch}}, please wait...
      </div>
    </div>

    <div class="result-foot">
      <pager></pager>
    </div>
  </div>
</template>

<script>
import SampleImage from './prediction_result_parts/sample_image.vue'
import Pager from './prediction_result_parts/pager.vue'

export default {
  name: "PredictionResult",
  components: {
    "sample-image": SampleImage,
    "pager": Pager,
  },
  computed: {
    getPredictResults: function () {
      return this.$store.getters.getPredictResults;
    },
  },
}
</script>

<style lang="scss" scoped>
#prediction-result {
  $component-margin-top: 32px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  width: 100%;
  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .result-head, .result-foot {
    display: flex;

    .title {
      line-height: $title-height;
      font-size: $title-font-size;
      font-weight: $font-weight-medium;
    }
    .page_nav {
      display: flex;
      margin-left: auto;
      margin-top: 16px;
    }
  }

  .content {
    display: flex;
    display: -webkit-flex;

    flex-flow: row wrap;

    width: 100%;

    background-color: $content-bg-color;
    border: 1px solid $content-border-color;
    border-radius: 4px;
  }
}

/* loading animation */
.loading {
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  width: 100vw;
  z-index: 999;
  background-color: #000000;
  opacity: 0.8;

  .animation {
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
  }
  .animation div {
    height:30px;
    width:10px;
    background-color: #ffffff;
    display:inline-block;
    margin-right:10px;
    -webkit-animation: animation-bar 1s infinite;
    animation: animation-bar 1s infinite;
  }

  .animation .bar1 {
    -webkit-animation-delay: 0.5s;
    animation-delay: 0.5s;
  }
  .animation .bar2 {
    -webkit-animation-delay: 0.6s;
    animation-delay: 0.6s;
  }
  .animation .bar3 {
    -webkit-animation-delay: 0.7s;
    animation-delay: 0.7s;
  }
  .animation .bar4 {
    -webkit-animation-delay: 0.8s;
    animation-delay: 0.8s;
  }
  .animation .bar5 {
    -webkit-animation-delay: 0.9s;
    animation-delay: 0.9s;
  }

  @keyframes animation-bar{
    30%
    {
        transform: scaleY(2);
    }
  }
}
.prediction-progress {
  position: absolute;
  top: 60%;
  left: 50%;
  -webkit-transform: translateX(-50%);
  transform: translateX(-50%);
  color: #000;
  opacity: 1;
}
</style>

