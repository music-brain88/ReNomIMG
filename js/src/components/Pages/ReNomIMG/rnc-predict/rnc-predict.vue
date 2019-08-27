<template>
  <div id="prediction-page">
    <rnc-prediction-panel-model-detail
      :width-weight="WidthWeightLeft"
      :height-weight="HeightWeight"
    />
    <rnc-prediction-panel-prediction-result
      :width-weight="WidthWeightRight"
      :height-weight="HeightWeight"
      @click-download="clickDownload()"
    />
  </div>
</template>

<script>
import RncPredictionPanelModelDetail from './../../../Organisms/rnc-prediction-panel-model-detail/rnc-prediction-panel-model-detail.vue'
import RncPredictionPanelPredictionResult from './../../../Organisms/rnc-prediction-panel-prediction-result/rnc-prediction-panel-prediction-result.vue'

export default {
  name: 'RncPredict',
  components: {
    'rnc-prediction-panel-prediction-result': RncPredictionPanelPredictionResult,
    'rnc-prediction-panel-model-detail': RncPredictionPanelModelDetail
  },
  data: function () {
    return {
      HeightWeight: 12,
      WidthWeightLeft: 3,
      WidthWeightRight: 9
    }
  },
  computed: {

  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
@import './../../../../../static/css/unified.scss';

#prediction-page {
  width: 100%;
  height: 100%;
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
}
</style>
