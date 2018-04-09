<template>
  <div id="model-detail-text">
    <div v-if="predictModel" class="model-content">
      <value-item :label="'Selected Model ID'" :val="predictModel.model_id"></value-item>

      <value-item :label="'Algorithm'" :val="getAlgorithmName(predictModel.algorithm)"></value-item>

      <value-item :label="'Hyper parameters - Total Epoch'" :val="predictModel.hyper_parameters['total_epoch']"></value-item>

      <value-item :label="'Hyper parameters - Batch Size'" :val="predictModel.hyper_parameters['batch_size']"></value-item>

      <value-item :label="'IoU'" :val="round(predictModel.best_epoch_iou, 100)*100 + '%'"></value-item>

      <value-item :label="'mAP'" :val="round(predictModel.best_epoch_map, 100)*100 + '%'"></value-item>

      <value-item :label="'Validation Loss'" :val="round(predictModel.validation_loss_list[predictModel.best_epoch], 1000)"></value-item>
    </div>
  </div>
</template>

<script>
import * as utils from '@/utils'
import * as constant from '@/constant'
import ValueItem from './value_item.vue'

export default {
  name: 'ModelDetailText',
  components: {
    'value-item': ValueItem
  },
  computed: {
    predictModel () {
      return this.$store.getters.getPredictModel
    }
  },
  methods: {
    getAlgorithmName: function (id) {
      return constant.ALGORITHM_NAME[id]
    },
    round: function (v, round_off) {
      const round_data = utils.round(v, round_off)
      if (Number.isNaN(round_data)) {
        return '-'
      } else {
        return round_data
      }
    }
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

