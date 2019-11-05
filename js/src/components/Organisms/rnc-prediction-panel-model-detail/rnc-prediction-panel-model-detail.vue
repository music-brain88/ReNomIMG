<template>
  <!-- <div class="rnc-prediction-panel-model-detail"> -->
  <rnc-title-frame
    id="rnc-prediction-panel-model-detail"
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Model Detail
      <div id="prediction-run-button">
        <rnc-button
          :button-label="'> Run Prediction'"
          :button-size-change="true"
          :disabled="!isRunnable"
          @click-button="runPredictionThread(model.id)"
        />
      </div>
    </template>
    <template slot="content-slot">
      <div
        v-if="model"
        id="deployed-model-datail"
      >
        <rnc-key-value
          :key-text="'Model ID :'"
          :value-text="model.id"
          class="item"
        />
        <rnc-key-value
          :key-text="'Algorithm :'"
          :value-text="getAlgorithmTitleFromId(model.algorithm_id)"
          class="item"
        />

        <div
          v-for="(param, key) in getAlgorithmParamList(model.algorithm_id)"
          :key="key"
        >
          <rnc-key-value
            :key-text="param.title + ' :'"
            :value-text="model.hyper_parameters[param.key]"
            class="item"
          />
        </div>
      </div>
    </template>
  </rnc-title-frame>
  <!-- </div> -->
</template>

<script>
import { mapGetters, mapActions } from 'vuex'
import RncKeyValue from './../../Atoms/rnc-key-value/rnc-key-value.vue'
import RncButton from './../../Atoms/rnc-button/rnc-button.vue'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'

export default {
  name: 'RncPredictionPanelModelDetail',
  components: {
    'rnc-key-value': RncKeyValue,
    'rnc-button': RncButton,
    'rnc-title-frame': RncTitleFrame
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 12,
    }
  },
  computed: {
    ...mapGetters([
      'getAlgorithmTitleFromId',
      'getAlgorithmParamList',
      'getDeployedModel'
    ]),
    model: function () {
      const model = this.getDeployedModel
      if (model) {
        return model
      } else {
        return false
      }
    },
    isRunnable () {
      return (this.model) && (this.model.isStopped())
    }
  },
  methods: {
    ...mapActions([
      'runPredictionThread'
    ])
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#prediction-run-button {
  height: 100%;
  width: 54%;
}

  #deployed-model-datail {
    height: 100%;
    width: 100%;
    padding: $padding-middle;

    .item {
      width: 100%;
      display: flex;
      margin-bottom: $margin-middle;
    }
}

</style>
