<template>
  <component-frame :width-weight="6" :height-weight="4">
    <template slot="header-slot">
      Model Detail
      <div id="deploy-button" @click="undeploy"
        v-if="getDeployedModel && getDeployedModel === getSelectedModel">
          <i class="fa fa-angle-right" aria-hidden="true"></i>&nbsp;Un Deploy
      </div>
      <div id="deploy-button" @click="deploy" v-else>
          <i class="fa fa-angle-right" aria-hidden="true"></i>&nbsp;Deploy
      </div>
    </template>
    <div id="model-detail">
      <div class="col" v-if="model">
        <div class="item">
          <div class="item-title">Model ID</div>
          <div class="item-content">{{ model.id }}</div>
        </div>
      </div>
      <div class="col" v-if="model">
        <div class="item">
          <div class="item-title">Algorithm</div>
          <div class="item-content">{{ getAlgorithmTitleFromId(model.algorithm_id) }}</div>
        </div>
        <div class="item">
          <div class="item-title">Dataset</div>
          <div class="item-content">{{ getDatasetName }}</div>
        </div>

        <div class="item"></div>

        <div class="item">
          <div class="item-title">{{ model.getResultOfMetric1().metric }}</div>
          <div class="item-content">{{ model.getResultOfMetric1().value }}</div>
        </div>
        <div class="item">
          <div class="item-title">{{ model.getResultOfMetric2().metric }}</div>
          <div class="item-content">{{ model.getResultOfMetric2().value }}</div>
        </div>
      </div>
      <div class="col" v-if="model">
        <div class="item" v-for="param in getAlgorithmParamList(model.algorithm_id)">
          <div class="item-title">{{ param.title }}</div>
          <div class="item-content">{{ model.hyper_parameters[param.key] }}</div>
        </div>
      </div>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import { ALGORITHM } from '@/const.js'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentModelDetail',
  components: {
    'component-frame': ComponentFrame
  },
  computed: {
    ...mapGetters([
      'getSelectedModel',
      'getAlgorithmTitleFromId',
      'getDatasetFromId',
      'getAlgorithmParamList',
      'getDeployedModel'
    ]),
    model: function () {
      const model = this.getSelectedModel
      if (model) {
        return model
      } else {
        return false
      }
    },
    getDatasetName: function () {
      const dataset = this.getDatasetFromId(this.model.dataset_id)
      if (dataset) {
        return dataset.name
      } else {
        return ''
      }
    }
  },
  created: function () {

  },
  methods: {
    ...mapMutations([
      'setDeployedModel',
      'showConfirm',
    ]),
    ...mapActions([
      'deployModel',
      'unDeployModel',
    ]),
    deploy: function () {
      const model = this.getSelectedModel
      const func = this.deployModel
      this.showConfirm({
        message: 'Are you sure deploy this model?',
        callback: function () { func(model) }
      })
    },
    undeploy: function () {
      const model = this.getDeployedModel
      const func = this.unDeployModel
      this.showConfirm({
        message: 'Are you sure undeploy this model?',
        callback: function () { func(model) }
      })
    }
  }
}
</script>

<style lang='scss'>

#deploy-button {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 25%;
  background-color: $component-header-sub-color;
  cursor: pointer;
  &:hover {
    background-color: $component-header-sub-color-hover;
  }
  &:active {
    background-color: $component-header-sub-color;
  }
}

#model-detail {
  width: 100%;
  padding: $total-model-padding-left-right;
  display: flex;
  .col {
    font-size: 90%;
    .item {
      width: calc(100% - #{$model-detail-item-margin-bottom});
      display: flex;
      align-items: center;
      margin-right: $model-detail-item-margin-right;
      margin-bottom: $model-detail-item-margin-bottom;
      .item-title {
        color: $component-font-color-title;
      }
      .item-content {
        text-align: center;
        color: $component-font-color;
        margin-left: 10px;
      }
    }
  }
}
</style>
