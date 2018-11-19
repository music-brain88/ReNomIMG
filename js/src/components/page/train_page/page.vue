<template>
  <div id="train-page">
    <div id="components">
      <component-progress/>
      <component-scatter/>
      <component-learning-curve/>
      <component-learning-curve/>
      <component-prediction-result/>
      <component-model-detail/>
      <component-tag-list/>
    </div>
    <div id="vertical-component">
      <component-model-filter/>
      <component-model-list/>
    </div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ComponentProgress from '@/components/page/train_page/component_progress.vue'
import ComponentScatter from '@/components/page/train_page/component_scatter.vue'
import ComponentModelList from '@/components/page/train_page/component_model_list.vue'
import ComponentModelFilter from '@/components/page/train_page/component_model_filter.vue'
import ComponentModelDetail from '@/components/page/train_page/component_model_detail.vue'
import ComponentTagList from '@/components/page/train_page/component_tag_list.vue'
import ComponentLearningCurve from '@/components/page/train_page/component_learning_curve.vue'
import ComponentPredictionResult from '@/components/page/train_page/component_prediction_result.vue'
import ComponentDataset from '@/components/page/dataset_page/component_dataset.vue'

export default {
  name: 'TrainPage',
  components: {
    'component-progress': ComponentProgress,
    'component-scatter': ComponentScatter,
    'component-tag-list': ComponentTagList,
    'component-model-detail': ComponentModelDetail,
    'component-model-list': ComponentModelList,
    'component-model-filter': ComponentModelFilter,
    'component-learning-curve': ComponentLearningCurve,
    'component-prediction-result': ComponentPredictionResult,
    'component-dataset': ComponentDataset,
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
#train-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  #components {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(10*#{$component-block-width});
  }  
  #vertical-component {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(2*#{$component-block-width});
  }
}
</style>
