<template>
  <div id="model-detail-text">
    <div v-if="predictModel" class="model-content">
      <value-item :label="'Selected Model ID'" :val="predictModel.model_id"></value-item>

      <value-item :label="'Algorithm'" :val="getAlgorithmNameById(predictModel.algorithm)"></value-item>

      <value-item :label="'Hyper parameters - Total Epoch'" :val="predictModel.hyper_parameters['total_epoch']"></value-item>

      <value-item :label="'Hyper parameters - Batch Size'" :val="predictModel.hyper_parameters['batch_size']"></value-item>

      <value-item :label="'IoU'" :val="predictModel.getRoundedIoU() + '%'"></value-item>

      <value-item :label="'mAP'" :val="predictModel.getRoundedMAP() + '%'"></value-item>

      <value-item :label="'Validation Loss'" :val="predictModel.getRoundedValidationLoss()"></value-item>
    </div>
  </div>
</template>

<script>
import ValueItem from './value_item.vue'

export default {
  name: "ModelDetailText",
  components: {
    "value-item": ValueItem,
  },
  computed: {
    predictModel() {
      return this.$store.getters.getPredictModel;
    }
  },
  methods: {
    getAlgorithmNameById: function(id) {
      return this.$store.getters.getAlgorithmNameById(id);
    },
  }
}
</script>

<style lang="scss" scoped>
#model-detail-text {
  $content-padding-top: 8px;
  $content-padding-bottom: 8px;
  $content-padding-horizontal: 16px;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  width: 100%;
  height: 100%;
  background-color: $content-bg-color;
  border: 1px solid $content-border-color;
  border-radius: 4px;

  .model-content {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;

    width: 100%;
    height: 100%;

    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;
  }
}
</style>

