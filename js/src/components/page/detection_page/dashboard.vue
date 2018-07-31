<template>
  <div id="dashboard">
    <div class="row">
      <div class="col-md-7 clear-padding dashboard_left">
        <div class="title">
          <div class="title-text">
            Dashboard
          </div>
        </div>
        <div class="content">
          <div class="dashboard-model-summary">
            <div>
              <model-ratio-bar></model-ratio-bar>
            </div>
            <div>
              <running-models></running-models>
            </div>
          </div>

        </div>
      </div>
      <div class="col-md-5 clear-padding dashboard_right">
        <div class="title">
          <div class="title-text">
            Model Map
          </div>
        </div>
        <div class="content">

          <div class="dashboard-model-plot">
            <model-plot></model-plot>
          </div>
        </div>

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
  $content-padding-top: 24px;
  $content-padding-horizontal: 24px;
  $content-padding-bottom: 16px;

  // height: $content-dashboard-height;
  margin: 0;
  margin-top: $component-margin-top;
  margin-left: $component-inner-horizontal-margin;

  .clear-padding{
    padding-right: 0;
  }

  .title {
    height:$content-top-header-hight;
    background-color: $header-color;
    font-size: $content-top-header-font-size;
    font-family: $content-top-header-font-family;
    color: $font-color;
    .title-text{
      line-height: $content-top-header-hight;
      margin-left: $content-top-heder-horizonral-margin;
    }
  }

  .content {
    display: flex;
    display: -webkit-flex;

    height: $content-dashboard-height;
    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;


    .dashboard-model-summary {
      width: 100%;
      #model-ratio-bar {
        height: 40%;
      }
      #running-models {
        height: 60%;
      }
    }

    .dashboard-model-plot {
      width: 100%;
      height: 100%;
      margin-left: 24px;
    }
  }
}
</style>
