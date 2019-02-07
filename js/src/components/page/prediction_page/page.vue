<template>
  <div id="prediction-page">
    <component-detail/>
    <component-prediction/>
  </div>
</template>

<script>
import ComponentPrediction from '@/components/page/prediction_page/component_prediction.vue'
import ComponentDetail from '@/components/page/prediction_page/component_detail.vue'

export default {
  name: 'PredictionPage',
  components: {
    'component-prediction': ComponentPrediction,
    'component-detail': ComponentDetail
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
#prediction-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  height: calc(#{$app-window-height} - #{$footer-height} - 70px);
}
</style>
