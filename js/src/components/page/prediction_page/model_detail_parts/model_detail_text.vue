<template>
  <div id="model-detail-text">
    <div v-if="predictModel" class="model-content">
      <value-item :label="'Selected Model ID'" :val="predictModel.model_id"></value-item>

      <value-item :label="'Dataset'" :val="predictModel.model_id"></value-item>

      <value-item :label="'Algorithm'" :val="getAlgorithmName(predictModel.algorithm)"></value-item>

      <value-item :label="'Total Epoch'" :val="predictModel.hyper_parameters['total_epoch']"></value-item>

      <value-item :label="'Batch Size'" :val="predictModel.hyper_parameters['batch_size']"></value-item>

      <value-item :label="'Image Width'" :val="predictModel.hyper_parameters['image_width']"></value-item>

      <value-item :label="'Image Height'" :val="predictModel.hyper_parameters['image_width']"></value-item>

      <component :is="additional_param_components[predictModel.algorithm_params]" :params="predictModel.algorithm_params"></component>

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
import Yolov1Params from '../../detection_page/model_detail_parts/yolov1_params.vue'
import Yolov2Params from '../../detection_page/model_detail_parts/yolov2_params.vue'

export default {
  name: 'ModelDetailText',
  components: {
    'value-item': ValueItem,
    'yolov1-params': Yolov1Params,
    'yolov2-params': Yolov2Params
  },
  data: function () {
    return {
      additional_param_components: ['yolov1-params', 'yolov2_params']
    }
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
    getDatasetName (dataset_def_id) {
      return this.$store.getters.getDatasetName
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


  .model-content {
    // display: flex;
    // flex-direction: column;
    // flex-wrap: wrap;
    //
    // width: 100%;
    // height: 100%;

    //padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;

    padding: calc(#{$content-padding-top}*4) $content-padding-horizontal $content-padding-bottom;
  }
}
</style>
