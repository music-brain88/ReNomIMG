<template>
  <div id="model-detail-text" class="row">

    <div class="col-md-3">
      <div class="title">
        Model ID {{ model.model_id }}
      </div>
    </div>
    <div class="col-md-4">

      <div class="model-values">
        <value-item :label="'Dataset'" :val="getDatasetName(model.dataset_def_id)"></value-item>
        <value-item :label="'Algorithm'" :val="getAlgorithmName(model.algorithm)"></value-item>
        <value-item :label="'Train Whole'" :val="Boolean(model.hyper_parameters['train_whole_network'])"></value-item>
        <value-item :label="'Total Epoch'" :val="model.hyper_parameters['total_epoch']"></value-item>
        <value-item :label="'Batch Size'" :val="model.hyper_parameters['batch_size']"></value-item>
        <value-item :label="'Image Width'" :val="model.hyper_parameters['image_width']"></value-item>
        <value-item :label="'Image Height'" :val="model.hyper_parameters['image_height']"></value-item>
        <!-- <value-item :label="'Seed'" :val="model.hyper_parameters['seed']"></value-item> -->
      </div>

    </div>
    <div class="col-md-5">

      <div class="additional_param">
        <component :is="additional_param_components[model.algorithm]" :params="model.algorithm_params"></component>
      </div>

      <value-item :label="'IoU'" :val="round_percent(model.best_epoch_iou) + '%'"></value-item>
      <value-item :label="'mAP'" :val="round_percent(model.best_epoch_map) + '%'"></value-item>
      <value-item :label="'Valid Loss'" :val="round(model.validation_loss_list[model.best_epoch], 1000)"></value-item>

    </div>
    <!-- <div class="title">
      Model ID {{ model.model_id }}
    </div> -->

    <!-- <div class="model-content"> -->
      <!-- <div class="model-values"> -->
        <!-- <value-item :label="'Dataset'" :val="getDatasetName(model.dataset_def_id)"></value-item>
        <value-item :label="'Algorithm'" :val="getAlgorithmName(model.algorithm)"></value-item>
        <value-item :label="'Train Whole'" :val="Boolean(model.hyper_parameters['train_whole_network'])"></value-item>
        <value-item :label="'Total Epoch'" :val="model.hyper_parameters['total_epoch']"></value-item>
        <value-item :label="'Batch Size'" :val="model.hyper_parameters['batch_size']"></value-item>
        <value-item :label="'Image Width'" :val="model.hyper_parameters['image_width']"></value-item>
        <value-item :label="'Image Height'" :val="model.hyper_parameters['image_height']"></value-item> -->
        <!-- <value-item :label="'Seed'" :val="model.hyper_parameters['seed']"></value-item> -->
      <!-- </div> -->

      <!-- <div class="model-values">
        <div class="additional_param">
          <component :is="additional_param_components[model.algorithm]" :params="model.algorithm_params"></component>
        </div>

        <value-item :label="'IoU'" :val="round_percent(model.best_epoch_iou) + '%'"></value-item>
        <value-item :label="'mAP'" :val="round_percent(model.best_epoch_map) + '%'"></value-item>
        <value-item :label="'Validation Loss'" :val="round(model.validation_loss_list[model.best_epoch], 1000)"></value-item>
      </div> -->
    <!-- </div> -->
  </div>
</template>

<script>
import * as utils from '@/utils'
import * as constant from '@/constant'
import ValueItem from './value_item.vue'
import Yolov1Params from './yolov1_params.vue'
import Yolov2Params from './yolov2_params.vue'

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
  props: {
    'model': {
      type: Object,
      required: true
    }
  },
  computed: {
    getDatasetName (dataset_def_id) {
      return this.$store.getters.getDatasetName
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
  // width: calc(100% - 12px);
  height: 100%;
  display: flex;
  .title {
    display: inline-flex;
    font-family: $content-inner-box-font-family;
    font-size: $content-inner-box-font-size;
  }

  .additional_param{
    padding-bottom: $content-bottom-padding;
  }

  .model-content {
    display: inline-flex;
    padding-left: $content-horizontal-padding;
    margin-bottom: $content-large-padding;
    .model-values {
      position: relative;
      width: 100%;
      padding-left: $content-horizontal-padding;

      .predict-model-selection-area {
        position: absolute;
        bottom: 0;
      }
    }
  }
}
</style>
