<template>
  <div id="prediction-page">
    <div class="model-detail-area">
      <model-detail></model-detail>
    </div>
    <div class="prediction-result-area">
      <prediction-result></prediction-result>
    </div>

    <image-modal v-if="$store.state.image_modal_show_flag"></image-modal>
  </div>
</template>

<script>
import ModelDetail from './model_detail.vue'
import PredictionResult from './prediction_result.vue'
import ImageModal from './image_modal.vue'

export default {
  name: 'PredictionPage',
  components: {
    'model-detail': ModelDetail,
    'prediction-result': PredictionResult,
    'image-modal': ImageModal
  },
  created: function () {
    this.$store.dispatch('initData', {'project_id': 1})
    this.$store.dispatch('loadDatasetInfov0')
  }
}
</script>

<style lang="scss" scoped>
#prediction-page {
  display: flex;
  display: -webkit-flex;
  flex-direction: column;
  -webkit-flex-direction: column;

  width: 100%;
  padding-bottom: 64px;

  .model-detail-area {
    width: 100%;
  }

  .prediction-result-area {
    width: 100%;
  }
}
</style>

