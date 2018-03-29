<template>
  <div id="model-detail">
    <div class="title">
      Model Detail
    </div>

    <div class="information" v-if="!$store.state.project || !$store.state.project.deploy_model_id">
      Please choose a model which you want to deploy from model list.
    </div>

    <div class="content">
      <div class="model-detail-area">
        <model-detail-text></model-detail-text>
      </div>

      <div class="button-area">
        <run-prediction-button></run-prediction-button>
        <export-button></export-button>
      </div>
    </div>
  </div>
</template>

<script>
import ModelDetailText from './model_detail_parts/model_detail_text.vue'
import RunPredictionButton from './model_detail_parts/run_prediction_button.vue'
import ExportButton from './model_detail_parts/export_button.vue'

export default {
  name: "ModelDetail",
  components: {
    "model-detail-text": ModelDetailText,
    "run-prediction-button": RunPredictionButton,
    "export-button": ExportButton,
  },
}
</script>

<style lang="scss" scoped>
#model-detail {
  $component-margin-top: 32px;

  $model-detail-height: 150px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  width: 100%;
  height: $model-detail-height;
  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .content {
    display: flex;
    display: -webkit-flex;

    width: 100%;
    height: calc(100% - #{$title-height});

    .model-detail-area {
      width: 84%;
      height: 100%;
    }

    .button-area {
      display: flex;
      flex-direction: column;
      position: relative;

      width: 16%;
      height: 100%;
    }
  }
}
</style>

