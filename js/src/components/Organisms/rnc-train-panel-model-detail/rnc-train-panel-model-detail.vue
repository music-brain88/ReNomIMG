<template>
  <rnc-title-frame
    id="rnc-train-panel-model-detail"
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Model Detail
      <div
        v-if="getDeployedModel && getSelectedModel && getDeployedModel.id === getSelectedModel.id"
        id="deploy-button"
        @click="undeploy"
      >
        <i
          class="fa fa-angle-right"
          aria-hidden="true"
        />
        &nbsp;Un Deploy
      </div>
      <div
        v-else
        id="deploy-button"
        @click="deploy"
      >
        <i
          class="fa fa-angle-right"
          aria-hidden="true"
        />
        &nbsp;Deploy
      </div>
    </template>
    <template slot="content-slot">
      <div id="model-detail">
        <div
          v-if="model"
          class="col"
        >
          <rnc-key-value
            :key-text="'Model ID'"
            :value-text="model.id"
            class="item"
          />
        </div>

        <div
          v-if="model"
          class="col"
        >
          <rnc-key-value
            :key-text="'Algorithm'"
            :value-text=" getAlgorithmTitleFromId(model.algorithm_id) "
            class="item"
          />
          <rnc-key-value
            :key-text="'Dataset'"
            :value-text=" getDatasetName "
            class="item"
          />

          <div class="item" />

          <rnc-key-value
            :key-text=" model.getResultOfMetric1().metric "
            :value-text=" model.getResultOfMetric2().value "
            class="item"
          />
        </div>

        <div
          v-if="model"
          class="col"
        >
          <rnc-key-value
            v-for="(param, key) in getAlgorithmParamList(model.algorithm_id)"
            :key="key"
            :key-text="param.title"
            :value-text="model.hyper_parameters[param.key]"
            class="item"
          />
        </div>
      </div>
    </template>
  </rnc-title-frame>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncKeyValue from './../../Atoms/rnc-key-value/rnc-key-value.vue'

export default {
  name: 'RncTrainPanelModelDetail',
  components: {
    'rnc-key-value': RncKeyValue,
    'rnc-title-frame': RncTitleFrame
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 6,
    }
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
  // watch: {
  //   getSelectedModel: function () {
  //     console.log('***getDeployedModel★★:' + this.getDeployedModel)
  //     console.dir(this.getDeployedModel)
  //     console.log('***getSelectedModel★★:' + this.getSelectedModel)
  //     console.dir(this.getSelectedModel)
  //   },
  //   getDeployedModel: function () {
  //     console.log('***getDeployedModel★★:' + this.getDeployedModel)
  //     console.dir(this.getDeployedModel)
  //     console.log('***getSelectedModel★★:' + this.getSelectedModel)
  //     console.dir(this.getSelectedModel)
  //   }
  // },
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
      this.$emit('click-deploy')
      const model = this.getSelectedModel
      const func = this.deployModel
      this.showConfirm({
        message: '<span style="line-height: 1.2rem;">' +
          'Are you sure to <span style="color: #f00;">deploy</span> this model?<br>' +
          '**This means undeploying current deployed model.</span>',
        callback: function () { func(model) }
      })
    },
    undeploy: function () {
      this.$emit('click-undeploy')
      const model = this.getDeployedModel
      const func = this.unDeployModel
      this.showConfirm({
        message: 'Are you sure to <span style="color: #f00;">undeploy</span> this model?',
        callback: function () { func(model) }
      })
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

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
  height: 100%;
  padding: $padding-middle;
  display: flex;
  justify-content: space-around;
  .col {
    height: 100%;
    font-size: 0.8rem;
    max-width: 50%;
    .item {
      width: 100%;
      text-align: center;
      line-height: $text-height-middle;
    }
  }
}

</style>
