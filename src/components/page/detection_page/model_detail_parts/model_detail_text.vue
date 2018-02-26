<template>
  <div id="model-detail-text">
    <div class="title">
      Model ID {{ modelData.model_id }}
    </div>

    <div class="model-content">
      <div class="model-values">
        <value-item :label="'Algorithm'" :val="getAlgorithmNameById(modelData.algorithm)"></value-item>
        <value-item :label="'Total Epoch'" :val="modelData.total_epoch"></value-item>
        <value-item :label="'Batch Size'" :val="modelData.hyper_parameters['batch_size']"></value-item>
        <value-item :label="'Image Width'" :val="modelData.hyper_parameters['image_width']"></value-item>
        <value-item :label="'Image Height'" :val="modelData.hyper_parameters['image_height']"></value-item>
        <br>
        <component :is="additional_param_components[modelData.algorithm]" :params="modelData.hyper_parameters['additional_params']"></component>
      </div>

      <div class="model-values">
        <value-item :label="'IoU'" :val="modelData.getRoundedIoU() + '%'"></value-item>
        <value-item :label="'mAP'" :val="modelData.getRoundedMAP() + '%'"></value-item>
        <value-item :label="'Validation Loss'" :val="modelData.getRoundedValidationLoss()"></value-item>
        <div class="predict-model-selection-area">
          <predict-model-selection :model="modelData"></predict-model-selection>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import ValueItem from './value_item.vue'
import YoloParams from './yolo_params.vue'
import PredictModelSelection from './predict_model_selection.vue'

export default {
  name: "ModelDetailText",
  components: {
    "value-item": ValueItem,
    "yolo-params": YoloParams,
    "predict-model-selection": PredictModelSelection,
  },
  data: function() {
    return {
      additional_param_components: ["yolo-params"],
    }
  },
  props: {
    "modelData": {
      type: Object,
      required: true
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

