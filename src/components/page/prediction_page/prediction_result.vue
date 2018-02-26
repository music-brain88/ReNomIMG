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
    <div id='loading' v-if='isPredicting'>
      <i id='spinner' class="fa fa-spinner fa-spin"></i>
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
    isPredicting: function () {
      return this.$store.state.predict_running_flag
    }
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

#loading {
  position: fixed;
  top:0;
  left:0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.5);
  color: #ffffff;
  #spinner {
    position:absolute;
    top:50%;
    left:50%;
    font-size: 4rem;
  }
}
</style>

