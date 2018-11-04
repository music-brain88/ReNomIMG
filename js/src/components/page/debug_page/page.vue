<!-->
  This is a page for debugging.
<-->
<template>
  <div id="debug_page">
    <component-progress/>
    <component-scatter/>
    <component-frame :width-weight="6" :height-weight="3">
      This is size(6, 4)
    </component-frame>
    <component-model-detail/>
    <component-prediction-result/>
    <component-tag-list/>
  </div>
</template>

<script>

import ComponentFrame from '@/components/common/component_frame.vue'
import ComponentProgress from '@/components/page/train_page/component_progress.vue'
import ComponentScatter from '@/components/page/train_page/component_scatter.vue'
import ComponentModelDetail from '@/components/page/train_page/component_model_detail.vue'
import ComponentTagList from '@/components/page/train_page/component_tag_list.vue'
import ComponentPredictionResult from '@/components/page/train_page/component_prediction_result.vue'

export default {
  name: 'DebugPage',
  components: {
    'component-frame': ComponentFrame,
    'component-progress': ComponentProgress,
    'component-scatter': ComponentScatter,
    'component-tag-list': ComponentTagList,
    'component-model-detail': ComponentModelDetail,
    'component-prediction-result': ComponentPredictionResult,
  },
  created: function () {

  }
}
</script>

<style lang="scss">
#debug_page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
}
</style>

