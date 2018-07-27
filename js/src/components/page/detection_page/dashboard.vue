<template>
  <div id="dashboard">
    <div class="title">
      Dashboard
    </div>

    <div class="content">
      <div class="dashboard-model-summary">
        <model-ratio-bar></model-ratio-bar>
        <running-models></running-models>
      </div>

      <div class="dashboard-model-plot">
        <model-plot></model-plot>
      </div>
    </div>
  </div>
</template>

<script>
import ModelRatioBar from './dashboard_parts/model_ratio_bar.vue'
import RunningModels from './dashboard_parts/running_models.vue'
import ModelPlot from './dashboard_parts/model_plot.vue'

export default {
  name: 'DashBoard',
  components: {
    'model-ratio-bar': ModelRatioBar,
    'running-models': RunningModels,
    'model-plot': ModelPlot
  }
}
</script>

<style lang="scss" scoped>
#dashboard {
  // $component-margin-top: 32px;
  //
  // $dashboard-height: 360px;
  //
  // $border-width: 2px;
  // $border-color: #006699;
  //
  // $title-height: 44px;
  // $title-font-size: 15pt;
  // $font-weight-medium: 500;
  //
  // $content-padding-top: 24px;
  // $content-padding-horizontal: 24px;
  // $content-padding-bottom: 16px;
  //
  // $content-bg-color: #ffffff;
  // $content-border-color: #cccccc;

  height: $dashboard-height;
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

    .dashboard-model-summary {
      width: 50%;
      #model-ratio-bar {
        height: 40%;
      }
      #running-models {
        height: 60%;
      }
    }

    .dashboard-model-plot {
      width: 50%;
      height: 100%;
      margin-left: 24px;
    }
  }
}
</style>
