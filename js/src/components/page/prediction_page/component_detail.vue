<template>
  <component-frame :width-weight="3" :height-weight="9">
    <template slot="header-slot">
      Model Detail
      <div id="prediction-run-button" :disabled="!model"
        @click="runPredictionThread(model.id)">
        <i class="fa fa-angle-right" aria-hidden="true"></i>&nbsp;Run Prediction
      </div>
    </template>
      <div id="deployed-model-datail" v-if="model">
        <div class="item">
          <div class="item-title">Model ID :</div>
          <div class="item-content">{{ model.id }}</div>
        </div>
        <div class="item">
          <div class="item-title">Algorithm :</div>
          <div class="item-content">{{ getAlgorithmTitleFromId(model.algorithm_id) }}</div>
        </div>
        <!--
        <div class="item">
          <div class="item-title">Dataset :</div>
          <div class="item-content">{{ getDatasetName }}</div>
        </div>
        -->
      <div class="item" v-for="param in getAlgorithmParamList(model.algorithm_id)">
        <div class="item-title">{{ param.title }} :</div>
        <div class="item-content">{{ model.hyper_parameters[param.key] }}</div>
      </div>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapActions } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentDetail',
  components: {
    'component-frame': ComponentFrame
  },
  computed: {
    ...mapGetters(['getDeployedModel',
      'getCurrentTask',
      'getTagColor',
      'getImagePageOfPrediction',
      'getAlgorithmTitleFromId',
      'getAlgorithmParamList',
    ]),
    model: function () {
      const model = this.getDeployedModel
      if (model) {
        return model
      } else {
        return false
      }
    }
  },
  created: function () {

  },
  methods: {
    ...mapActions(['runPredictionThread'])
  }
}
</script>

<style lang='scss'>


#prediction-run-button {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 45%;
  background-color: $component-header-sub-color;
  cursor: pointer;
}

#deployed-model-datail {
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  .item {
    width: calc(100% - #{$model-detail-item-margin-bottom}*2);
    display: flex;
    margin: $model-detail-item-margin-bottom;
    border-bottom: solid 1px lightgray;
    .item-title {
      width: 60%;
      color: $model-detail-item-title-font-color;
    }
    .item-content {
      width: 40%;
      text-align: center;
    }
  }
}
</style>
