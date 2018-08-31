<template>
  <div id="model-detail">
    <div class="row">
      <div class="col-md-7 col-sm-12 clear-padding">
        <div class="title">
          <div class="row">
            <div class="col-md-9 col-sm-12">
              <div class="title-text">
                Model Detail
              </div>
            </div>
            <div class="col-md-3 col-sm-12" v-bind:class="{'panel-selected':model,'panel-not-selected': !model}">
              <predict-model-selection v-if="model" :model="model"></predict-model-selection>
            </div>
          </div>
        </div>
        <div class="content">
          <div class="model-detail-text">
            <model-detail-text v-if="model" :model="model"></model-detail-text>
          </div>
        </div>
      </div>
      <div class="col-md-5 col-sm-12 clear-padding">
        <div class="title">
          <div class="title-text">
            Learning Curve
          </div>
        </div>
        <div class="content">
          <div class="model-detail-learning-curve">
            <learning-curve v-if='model'
              :trainLoss="model.train_loss_list"
              :validationLoss="model.validation_loss_list"></learning-curve>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import ModelDetailText from './model_detail_parts/model_detail_text.vue'
import LearningCurve from './model_detail_parts/learning_curve.vue'
import PredictModelSelection from './model_detail_parts/predict_model_selection.vue'

export default {
  name: 'ModelDetail',
  components: {
    'model-detail-text': ModelDetailText,
    'learning-curve': LearningCurve,
    'predict-model-selection': PredictModelSelection
  },
  computed: {
    model () {
      return this.$store.getters.getSelectedModel
    }
  }
}
</script>

<style lang="scss" scoped>
#model-detail {

  margin: 0;
  margin-top: $component-margin-top;
  // margin-left: $component-inner-horizontal-margin;

  .title {
    height:$content-top-header-hight;
    font-size: $content-top-header-font-size;
    font-family: $content-top-header-font-family;
    background:$header-color;
    color:$font-color;
    .title-text{
      line-height: $content-top-header-hight;
      margin-left: $content-top-heder-horizonral-margin;
    }
    .panel-selected{
      background-color: $panel-bg-color;
      &:hover{
        background-color: $panel-bg-color-hover;
      }
    }

    .panel-not-selected{
      background: $panel-bg-color;
    }
  }

  .content {
    height: $content-detail-height;
    margin-top: $content-top-margin;
    display: flex;
    display: -webkit-flex;

    // min-height:calc(100% - #{$content-top-header-hight});
    padding: $content-top-padding $content-horizontal-padding $content-bottom-padding;

    background-color: $content-bg-color;
    border: 1px solid $content-border-color;

    .model-detail-text {
      width: 100%;
      height: 100%;
    }
    &:after{
      content:'';
    }

    .model-detail-learning-curve {
      width: 100%;
      height: 100%;
    }
  }
}
</style>
