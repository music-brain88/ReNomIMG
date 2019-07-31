<template>
  <rnc-title-frame
    id="rnc-train-panel-progress"
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Train Progress
    </template>
    <template slot="content-slot">
      <div id="component-progress-model-num">
        <div id="model-bar-legend">
          <div id="total-num">
            Total Models : {{ getFilteredModelList.length }}
          </div>
          <div id="legend">
            <rnc-color-label
              v-for="(alg, key) in getAlgorithList()"
              id="legend-item"
              :title="alg.title"
              :color-class="getAlgColorClass(alg.id)"
              :key="key"
            />
            <rnc-color-label
              id="legend-item"
              :title="'Reserved'"
              :color-class="'color-reserved'"
            />
            <rnc-color-label
              id="legend-item"
              :title="'Created'"
              :color-class="'color-created'"
            />
          </div>
        </div>

        <rnc-bar-model
          :model-info="reduceModelList(getFilteredModelList)"
          :style="{ width: '100%', height: '7%'}"
        />

        <div
          id="component-progress"
          class="scrollbar-container"
        >
          <div id="progress-title">
            Running Progress
          </div>
          <div id="progress-bars">
            <rnc-progress-detail
              v-if="getRunningModelList.length > 0"
              :is-title="true"
            />
            <rnc-progress-detail
              v-for="(item, key) in getRunningModelList"
              :key="key"
              :model="item"
              :color-class="getColorClass(item)"
              @click-stop-button="callModal"
            />
          </div>
        </div>
      </div>
    </template>
  </rnc-title-frame>
</template>

<script>

import { mapGetters, mapMutations, mapActions } from 'vuex'
import { TASK_ID, ALGORITHM } from '../../../const.js'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncBarModel from './../../Atoms/rnc-bar-model/rnc-bar-model.vue'
import RncProgressDetail from './../../Molecules/rnc-progress-detail/rnc-progress-detail.vue'
import RncColorLabel from './../../Atoms/rnc-color-label/rnc-color-label.vue'

const RESERVED = '-1'
const CREATED = '-2'

export default {
  name: 'RncTrainPanelProgress',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-progress-detail': RncProgressDetail,
    'rnc-bar-model': RncBarModel,
    'rnc-color-label': RncColorLabel
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 5,
    },
  },
  data: function () {
    return {
      onHovering: false
    }
  },
  computed: {
    ...mapGetters([
      'getRunningModelList',
      'getFilteredModelList',
      // 'getAlgorithmColor',
      'getCurrentTask',
      'getColorClass',
    ]),
  },
  created: function () {
    window.addEventListener('resize', this.draw, false)
  },
  methods: {
    ...mapActions(['stopModelTrain']),
    ...mapMutations(['showConfirm']),

    callModal: function (model_id) {
      const func = this.stopModelTrain
      this.showConfirm({
        message: "Are you sure you want to <span style='color: #f00;}'>stop</span> this model?",
        callback: function () { func(model_id) }
      })
    },
    reduceModelList: function (model_list) {
      model_list = Object.entries(model_list.reduce(
        function (algs, model) {
          var id = 0
          if (model.isReserved()) {
            id = RESERVED
          } else if (model.isCreated()) {
            id = CREATED
          } else {
            id = model.algorithm_id
          }
          if (id in algs) {
            algs[id] += 1
          } else {
            algs[id] = 1
          }
          return algs
        }, {})).map(d => [d[0], parseFloat(d[1]) / parseFloat(model_list.length), d[1]])
      return model_list
    },

    /*
     * Get all of algorithm info for the currnt task by refering const.js 'ALGORITHM'
     */
    getAlgorithList: function () {
      const task = this.getCurrentTask
      let arr
      if (task === TASK_ID.CLASSIFICATION) {
        arr = Object.values(ALGORITHM.CLASSIFICATION)
      } else if (task === TASK_ID.DETECTION) {
        arr = Object.values(ALGORITHM.DETECTION)
      } else if (task === TASK_ID.SEGMENTATION) {
        arr = Object.values(ALGORITHM.SEGMENTATION)
      }
      return arr.map(d => { return { title: d.title, key: d.key, id: d.id } })
    },

    /*
     * Get algorithm colors by checking the objects obtained in getAlgorithmList.
     */
    getAlgColorClass: function (alg_id) {
      const id = alg_id % 10
      return 'color-' + id
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

#component-progress-model-num {
  width:calc(100% - 2 * $padding-extra-large);
  height: 100%;
  display: flex;
  flex-wrap: wrap;
  padding: 0 $padding-extra-large;
  #model-bar-legend {
    width: 100%;
    height: 30%;
    #total-num {
      width: 100%;
      height: 80%;
      display: flex;
      align-items: flex-end;
      color: $component-font-color-title;
    }
    #legend {
      width: 100%;
      height: 20%;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-end;
      #legend-item {
        height: 100%;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        margin-left: $margin-middle;
        #legend-box {
          width: $total-model-legend-box-width;
          height: $total-model-legend-box-width;
        }
        #legend-title {
          font-size: $component-font-size-small;
          margin-left: $total-model-legend-box-margin-left;
        }
      }
    }
  }
  #model-bar {
    width: 100%;
    height: 7%;
    display: flex;
    overflow: hidden;
    section {
      height: 100%;
      min-width: 10%;
      line-height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      color: white;
      #alg-title {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 55%;
      }
    }
  }
  #component-progress {
    width: 100%;
    height: calc(63% - #{$progress-bar-area-margin-top});
    overflow: auto;
    line-height: auto;
    margin-top: $progress-bar-area-margin-top;
    #progress-title {
      width: 100%;
      margin-bottom: $total-model-title-padding-bottom;
      color: $component-font-color-title;
      #progress-bars {
        width: 100%;
      }
    }
  }
  #green {
    background: #229954;
  }
}
</style>
