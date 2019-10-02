<template>
  <div id="train-page">
    <div id="components">
      <rnc-train-panel-progress
        :width-weight="7"
        :height-weight="5.5"
      />
      <rnc-train-panel-model-scatter
        :width-weight="5"
        :height-weight="5.5"
      />
      <rnc-train-panel-model-detail
        :width-weight="7"
        :height-weight="5.5"
      />
      <rnc-train-panel-learning-curve
        :width-weight="5"
        :height-weight="5.5"
      />
      <rnc-train-panel-prediction-result
        :width-weight="9"
        :height-weight="7"
      />
      <rnc-train-panel-tag-list
        :width-weight="3"
        :height-weight="7"
      />
    </div>
    <div id="vertical-component">
      <rnc-train-panel-model-list
        :width-weight="12"
        :height-weight="17"
      />
    </div>
  </div>
</template>

<script>
import RncTrainPanelProgress from '../../../Organisms/rnc-train-panel-progress/rnc-train-panel-progress.vue'
import RncTrainPanelModelScatter from '../../../Organisms/rnc-train-panel-model-scatter/rnc-train-panel-model-scatter.vue'
import RncTrainPanelModelDetail from '../../../Organisms/rnc-train-panel-model-detail/rnc-train-panel-model-detail.vue'
import RncTrainPanelLearningCurve from '../../../Organisms/rnc-train-panel-learning-curve/rnc-train-panel-learning-curve.vue'
import RncTrainPanelPredictionRusult from '../../../Organisms/rnc-train-panel-prediction-result/rnc-train-panel-prediction-result.vue'
import RncTrainPanelTagList from '../../../Organisms/rnc-train-panel-tag-list/rnc-train-panel-tag-list.vue'
import RncTrainPanelModelList from '../../../Organisms/rnc-train-panel-model-list/rnc-train-panel-model-list.vue'

export default {
  name: 'RncTrain',
  components: {
    'rnc-train-panel-progress': RncTrainPanelProgress,
    'rnc-train-panel-model-scatter': RncTrainPanelModelScatter,
    'rnc-train-panel-model-detail': RncTrainPanelModelDetail,
    'rnc-train-panel-learning-curve': RncTrainPanelLearningCurve,
    'rnc-train-panel-prediction-result': RncTrainPanelPredictionRusult,
    'rnc-train-panel-tag-list': RncTrainPanelTagList,
    'rnc-train-panel-model-list': RncTrainPanelModelList
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

#train-page {
  background-color: #f4f4f2;
  height: 100%;
  width: 100%;
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  #components {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    height: 100%;
    width: calc(9.5 * #{$component-block-width});
  }
  #vertical-component {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(2.5 * #{$component-block-width});
  }
}
</style>
