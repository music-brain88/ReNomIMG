<template>
  <div id="model-detail">
    <div class="add-panel">
      <run-prediction-button></run-prediction-button>
    </div>
    <div class="title-selectbox">
      <div class="title">
        Model Detail
      </div>
    </div>


    <div class="content">
      
      <div class="row">
        <div class="col-md-12">
          <div class="information" v-if="!$store.state.project || !$store.state.project.deploy_model_id">
            Please choose a model which you want to deploy from model list.
          </div>
          <model-detail-text></model-detail-text>
        </div> 
      </div>
      <div class="row">
        <div class="col-md-12">
          <export-button></export-button>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
import ModelDetailText from './model_detail_parts/model_detail_text.vue'
import RunPredictionButton from './model_detail_parts/run_prediction_button.vue'
import ExportButton from './model_detail_parts/export_button.vue'

export default {
  name: 'ModelDetail',
  components: {
    'model-detail-text': ModelDetailText,
    'run-prediction-button': RunPredictionButton,
    'export-button': ExportButton
  }
}
</script>

<style lang="scss" scoped>

#model-detail {

  width: 100%;
  margin: 0;
  margin-top: $component-margin-top;

  .add-panel {
    height: $panel-height;
    background-color: $panel-bg-color;
    color: $font-color;
    font-size: $content-top-header-font-size;
    text-align: center;
    line-height: $panel-height;
    cursor: pointer;
  }
  .add-panel:hover {
    background-color: $panel-bg-color-hover;
  }

  .title-selectbox {
    display: flex;
    display: -webkit-flex;
    -webkit-justify-content: space-between;
    justify-content: space-between;
    color: $font-color;
    background: $header-color;
    margin-top:$content-top-margin;
  }

  .title {
    line-height: $content-top-header-hight;
    font-family:$content-top-header-font-family;
    font-size:$content-top-header-font-size;
    margin-left: $content-top-heder-horizonral-margin;
  }

  .content {
    margin-top:$content-top-margin;
    width: 100%;
    height: calc(#{$content-predictionpage-content-predict-result-height} - (#{$content-top-header-hight} + #{$content-top-margin}));
    background:$content-bg-color;
    padding: $content-top-padding 0 $content-bottom-padding;

  }
}
</style>
