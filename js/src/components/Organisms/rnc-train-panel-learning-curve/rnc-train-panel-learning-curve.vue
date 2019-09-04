<template>
  <!-- <div class="rnc-train-panel-learning-curve"> -->
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Learning Curve
    </template>

    <template slot="content-slot">
      <div class="learning-curve">
        <div class="legend">
          <rnc-color-label
            :title=" 'Train' "
            :color-class=" 'color-train' "
            :is-toggle-label="true"
            :line-through="switchTrainGraph"
            class="label"
            @click="switch_train_graph()"
          />
          <rnc-color-label
            :title=" 'Valid' "
            :color-class=" 'color-valid' "
            :is-toggle-label="true"
            :line-through="switchValidGraph"
            class="label"
            @click="switch_valid_graph()"
          />
          <div class="best-epoch">
            <span class="legend-line">
              &mdash;
            </span>
            Best Epoch Line
          </div>
        </div>

        <rnc-grid-x-y
          :selected-model-obj="SelectedModelObj"
          :switch-train-graph="switchTrainGraph"
          :switch-valid-graph="switchValidGraph"
          :end-of-axis-x-y="endOfAxisXY"
          kind="learning-curve"
          axis-name-x="Epoch [-]"
          axis-name-y="Loss [-]"
        />
      </div>
    </template>
  </rnc-title-frame>
  <!-- </div> -->
</template>

<script>
import { mapGetters } from 'vuex'
import RncColorLabel from './../../Atoms/rnc-color-label/rnc-color-label.vue'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncGridXY from './../../Molecules/rnc-grid-x-y/rnc-grid-x-y.vue'

export default {
  name: 'RncTrainPanelLearningCurve',
  components: {
    'rnc-color-label': RncColorLabel,
    'rnc-title-frame': RncTitleFrame,
    'rnc-grid-x-y': RncGridXY
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 5,
    }
  },
  data: function () {
    return {
      // AxisNameX: "Epoch [-]",
      // AxisNameY: "Loss [-]",
      SelectedModelObj: {},
      switchTrainGraph: false,
      switchValidGraph: false,
      endOfAxisXY: {
        'x': {
          'max': 10,
          'min': 0
        },
        'y': {
          'max': 10,
          'min': 0
        }
      }
    }
  },
  computed: {
    ...mapGetters([
      'getSelectedModel'
    ])
  },
  watch: {
    getSelectedModel: function () {
      this.SelectedModelObj = this.getSelectedModel
      this.shapeAxisXY()
    }
  },
  mounted: function () {
    this.SelectedModelObj = this.getSelectedModel
    this.shapeAxisXY()
  },
  methods: {
    switch_train_graph: function () {
      this.switchTrainGraph = !this.switchTrainGraph
    },
    switch_valid_graph: function (flg) {
      this.switchValidGraph = !this.switchValidGraph
    },
    shapeAxisXY: function () {
      if (!this.getSelectedModel) return

      const train_loss_list = this.getSelectedModel.train_loss_list
      const valid_loss_list = this.getSelectedModel.valid_loss_list
      const learning_epoch = train_loss_list.length
      let maxX = Math.max(learning_epoch + 1, 10)
      maxX = Math.ceil(maxX / 5) * 5
      const minX = 0
      let maxY = Math.max((Math.max.apply(null, [...train_loss_list, ...valid_loss_list]) * 1.1), 1)
      maxY = Math.ceil(maxY)
      let minY = Math.min(Math.min.apply(null, [...train_loss_list, ...valid_loss_list]), 0)
      minY = Math.floor(minY)

      this.endOfAxisXY = {
        'x': {
          'max': maxX,
          'min': minX
        },
        'y': {
          'max': maxY,
          'min': minY
        }
      }
      // console.log('変換後endOfAxisXY：', this.endOfAxisXY)
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.learning-curve {
  width: 100%;
  height: 100%;
  position: relative;
  color: $gray;
  .legend {
    position: absolute;
    width: calc(100% - #{$scatter-padding}*2);
    top: 20px;
    display: flex;
    flex-wrap: nowrap;
    justify-content: flex-end;
    align-items: center;
    font-size: $fs-small;
    .graph-off {
      text-decoration: line-through;
    }
    .label {
      z-index: 10;
      margin-right: 9px;
    }
    .best-epoch {
      .legend-line {
        display: inline-block;
        width: 10px;
        margin-left: 4px;
        margin-right: 4px;
        font-size: 10px;
        font-weight: bold;
        color: $red;
      }
    }
  }
}
</style>
