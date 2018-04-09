<template>
  <div id="model-detail">
    <div class="title">
      Model Detail
    </div>

    <div class="content">
      <div class="model-detail-text">
        <model-detail-text v-if="model" :model="model"></model-detail-text>
      </div>
      <div class="model-detail-learning-curve">
        <learning-curve v-if='model'
          :trainLoss="model.train_loss_list"
          :validationLoss="model.validation_loss_list"></learning-curve>
      </div>
    </div>
  </div>
</template>

<script>
import ModelDetailText from './model_detail_parts/model_detail_text.vue'
import LearningCurve from './model_detail_parts/learning_curve.vue'

export default {
  name: 'ModelDetail',
  components: {
    'model-detail-text': ModelDetailText,
    'learning-curve': LearningCurve
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
  $component-margin-top: 32px;

  $model-detail-height: 360px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-padding-top: 24px;
  $content-padding-horizontal: 24px;
  $content-padding-bottom: 16px;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

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

    height: calc(100% - #{$title-height});
    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;

    background-color: $content-bg-color;
    border: 1px solid $content-border-color;
    border-radius: 4px;

    .model-detail-text {
      width: 50%;
      height: 100%;
    }

    .model-detail-learning-curve {
      width: 50%;
      height: 100%;
      margin-left: 24px;
    }
  }
}
</style>

