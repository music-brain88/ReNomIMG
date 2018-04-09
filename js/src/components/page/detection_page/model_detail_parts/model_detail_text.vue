<template>
  <div id="model-detail-text">
    <div class="title">
      Model ID {{ model.model_id }}
    </div>

    <div class="model-content">
      <div class="model-values">
        <value-item :label="'Algorithm'" :val="getAlgorithmName(model.algorithm)"></value-item>
        <value-item :label="'Total Epoch'" :val="model.hyper_parameters['total_epoch']"></value-item>
        <value-item :label="'Batch Size'" :val="model.hyper_parameters['batch_size']"></value-item>
        <value-item :label="'Image Width'" :val="model.hyper_parameters['image_width']"></value-item>
        <value-item :label="'Image Height'" :val="model.hyper_parameters['image_height']"></value-item>
        <!-- <value-item :label="'Seed'" :val="model.hyper_parameters['seed']"></value-item> -->
        <br>
        <component :is="additional_param_components[model.algorithm]" :params="model.algorithm_params"></component>
      </div>

      <div class="model-values">
        <value-item :label="'IoU'" :val="round_percent(model.best_epoch_iou) + '%'"></value-item>
        <value-item :label="'mAP'" :val="round_percent(model.best_epoch_map) + '%'"></value-item>
        <value-item :label="'Validation Loss'" :val="round(model.validation_loss_list[model.best_epoch], 1000)"></value-item>
        <div class="predict-model-selection-area">
          <predict-model-selection :model="model"></predict-model-selection>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as utils from '@/utils'
import * as constant from '@/constant'
import ValueItem from './value_item.vue'
import YoloParams from './yolo_params.vue'
import PredictModelSelection from './predict_model_selection.vue'

export default {
  name: 'ModelDetailText',
  components: {
    'value-item': ValueItem,
    'yolo-params': YoloParams,
    'predict-model-selection': PredictModelSelection
  },
  data: function () {
    return {
      additional_param_components: ['yolo-params']
    }
  },
  props: {
    'model': {
      type: Object,
      required: true
    }
  },
  methods: {
    getAlgorithmName: function (algorithm) {
      return constant.ALGORITHM_NAME[algorithm]
    },
    round: function (v, round_off) {
      if (v == null) {
        return '-'
      } else {
        const round_data = utils.round(v, round_off)
        return round_data
      }
    },
    round_percent: function (v) {
      if (v == null) {
        return '-'
      } else {
        return utils.round_percent(v)
      }
    }
  }
}
</script>

<style lang="scss" scoped>
#model-detail-text {
  $title-height: 24px;
  $title-font-size: 16px;
  $font-weight-medium: 500;

  $content-margin: 8px;

  width: calc(100% - 12px);
  height: 100%;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
    border-bottom: 1px solid #ccc;
  }

  .model-content {
    display: flex;

    margin-bottom: $content-margin;
    .model-values {
      position: relative;
      width: 50%;
      margin-right: $content-margin;
      padding-top: $content-margin;

      .predict-model-selection-area {
        position: absolute;
        bottom: 0;
      }
    }
  }
}
</style>

