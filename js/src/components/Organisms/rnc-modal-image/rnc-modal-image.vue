<template>
  <div
    id="image-modal"
    ref="container"
    tabindex="0"
    @keyup.right="nextPage"
    @keyup.left="prevPage"
  >
    <rnc-title-frame
      id="image-result"
      :width-weight="7.5"
      :height-weight="6.5"
    >
      <template slot="header-slot">
        Prediction Result &nbsp;&nbsp; {{ modal_index+1 }} / {{ length }}
        <div id="checkbox-area">
          <rnc-button-switch
            :checked="show_prediction"
            :label="'Prediction'"
            class="margin-right"
            @change="onChangePredictionCheckBox"
          />
          <rnc-button-switch
            :checked="show_target"
            :label="'Target'"
            class="margin-right"
            @change="onChangeTargetCheckBox"
          />
        </div>
      </template>
      <template slot="content-slot">
        <div
          id="image-container"
          ref="imageContainer"
        >
          <rnc-labeled-image
            :show-target="show_target"
            :show-predict="show_prediction"
            :show-image="show_image"
            :img="img"
            :width="size[0]"
            :height="size[1]"
            :max-width="canvas_width"
            :max-height="canvas_height"
            :model="getSelectedModel"
            :result="getResult()"
            :dataset="dataset"
          />
        </div>
      </template>
    </rnc-title-frame>

    <rnc-title-frame
      v-if="isTaskClassification || isTaskDetection"
      id="result"
      :width-weight="4.5"
      :height-weight="6.5"
    >
      <template slot="header-slot">
        <div class="result-header">
          <span>
            No.
          </span>
          <span>
            Name
          </span>
          <span>
            Score
          </span>
        </div>
      </template>
      <template slot="content-slot">
        <div id="result-container">
          <div
            v-if="isTaskClassification"
            id="cls-result"
            class="result"
          >
            <div
              v-for="(item, index) in getClassificationTop3"
              :key="index"
            >
              <span>
                {{ index + 1 }}
              </span>
              <span>
                {{ item.index }}
              </span>
              <span>
                {{ item.score }}%
              </span>
            </div>

            <div v-if="!getClassificationTop3">
              <span />
              <span>
                No prediction
              </span>
              <span />
            </div>
          </div>
          <div
            v-else-if="isTaskDetection"
            id="box-result"
            class="result"
          >
            <div
              v-for="(r, index) in getPredictedBox"
              :key="index"
            >
              <span>
                {{ index }}
              </span>
              <span>
                {{ r.name }}
              </span>
              <span>
                {{ r.score.toFixed(2) }}
              </span>
            </div>
            <div v-if="getPredictedBox.length === 0">
              <span />
              <span>
                No Prediction
              </span>
              <span />
            </div>
          </div>

          <div
            v-else-if="isTaskSegmentation"
            id="seg-result"
            class="result"
          />
        </div>
      </template>
    </rnc-title-frame>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncButtonSwitch from './../../Atoms/rnc-button-switch/rnc-button-switch.vue'
import RncLabeledImage from './../../Molecules/rnc-labeled-image/rnc-labeled-image.vue'

export default {
  name: 'RncModalImage',
  components: {
    'rnc-labeled-image': RncLabeledImage,
    'rnc-title-frame': RncTitleFrame,
    'rnc-button-switch': RncButtonSwitch
  },
  data: function () {
    return {
      hoverBox: null,
      /**
        The states of checkbox. Allowed patterns.

          Classification:
                 show_image: disabled  disabled
                show_target:   true     false
            show_prediction:  false      true

          Detection:
                 show_image: disabled  disabled  disabled  disabled
                show_target:   true     false      true     false
            show_prediction:   true     false     false      true

          Segmentation:
                 show_image:   true      true     false    false
                show_target:   true     false      true    false
            show_prediction:  false      true     false     true
      */
      show_image: true, // Show image or not.
      show_target: false, // Show target or not.
      show_prediction: true, // Show prediction result or not.
      canvas_width: 0,
      canvas_height: 0,
    }
  },
  computed: {
    ...mapState([
      'datasets',
      'modal_image',
      'modal_index',
      'datasets'
    ]),
    ...mapGetters([
      'getSelectedModel',
      'getCurrentTask',
      'isTaskClassification',
      'isTaskDetection',
      'isTaskSegmentation',
    ]),
    model: function () {
      const model = this.getSelectedModel
      if (model) {
        return model
      }
      return null
    },
    dataset: function () {
      const model = this.model
      if (model) {
        const dataset = this.datasets.find(d => d.id === model.dataset_id)
        if (dataset) {
          return dataset
        }
      }
      return null
    },
    class_map: function () {
      const map = this.dataset.class_map
      return map
    },
    img: function () {
      const index = this.modal_index
      return this.dataset.valid_data.img[index]
    },
    size: function () {
      const index = this.modal_index
      return this.dataset.valid_data.size[index]
    },
    length: function () {
      return this.dataset.valid_data.img.length
    },
    getClassificationTop3: function () {
      const model = this.model
      const map = this.class_map
      if (!model) return
      const prediction = model.getValidResult(this.modal_index)
      if (!prediction) return
      const score = prediction.score.map((s, index) => {
        return {
          index: map[index],
          score: (s * 100)
        }
      })
      const top5 = score.sort((a, b) => {
        if (a.score < b.score) {
          return 1
        } else {
          return -1
        }
      }).slice(0, 5)
      return top5.map(d => { return { index: d.index, score: d.score.toFixed(2) } })
    },
    getPredictedBox: function () {
      const model = this.model
      if (!model) return []
      // TODO muraishi: input a data with ValidResult so the it could be shown
      // TODO muraishi : to show getPredictedBox, model has to contain value: "best_epoch_valid_result.prediction"
      const prediction = model.getValidResult(this.modal_index)
      return (prediction) || []
    }
  },
  mounted: function () {
    this.$refs.container.focus()
    const el = this.$refs.imageContainer
    this.canvas_width = el.clientWidth
    this.canvas_height = el.clientHeight
  },

  methods: {
    ...mapMutations(['setImageModalData']),
    ...mapActions(['loadSegmentationTargetArray']),
    onChangePredictionCheckBox: function (e) {
      this.show_prediction = e.target.checked
      this.show_target = (!this.show_prediction || this.isTaskDetection) && this.show_target
    },
    onChangeTargetCheckBox: function (e) {
      this.show_target = e.target.checked
      this.show_prediction = (!this.show_target || this.isTaskDetection) && this.show_prediction
    },
    nextPage: function () {
      this.setImageModalData(Math.min(this.length - 1,
        this.modal_index + 1))
      this.hoverBox = null
    },
    prevPage: function () {
      this.setImageModalData(Math.max(0, this.modal_index - 1))
      this.hoverBox = null
    },
    getResult: function () {
      const index = this.modal_index
      const model = this.getSelectedModel
      const dataset = this.dataset
      if (!model || !dataset) return
      const pred = model.getValidResult(index)
      const targ = dataset.getValidTarget(index)
      return {
        index: index,
        target: targ,
        predict: pred
      }
    },
  },
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#image-modal {
  width: 100%;
  height: 100%;
  display: flex;
  padding: 10px;
  #image-result {
    .component-header {
      #checkbox-area {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        height: 100%;
        width: 50%;
        .margin-right{
          margin-right: 10px;
        }
      }
    }
    .frame-content{
      #image-container {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: calc(100% - 40px - 10px);
      }
    }
  }

  #result {
    .component-header {
      .result-header {
        width: 100%;
        color: white;
        display: flex;
        span {
          &:nth-child(1) {
            width: 32%;
          }
          &:nth-child(2) {
            width: 35%;
          }
        }
      }
    }
    .frame-content{
      #result-container {
        width: 100%;
        height: calc(100% - 40px - 10px);
        #cls-result div{
          display: flex;
          width: 100%;
          height: 9%;
          border-bottom: solid 1px lightgray;
          &:first-child {
            font-size: 1.1rem;
          }
          span {
            display: flex;
            align-items: center;
            justify-content: space-around;
            width: 33.3%;
          }
        }
        #box-result {
          overflow: auto;
        }
        #box-result div {
          display: flex;
          width: 100%;
          height: 9%;
          border-bottom: solid 1px lightgray;
          justify-content: space-between;;
          span {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            width: 33.3%;
            padding-left: 20px;
            &:nth-child(1) {
              width: 25%;
            }
            &:nth-child(2) {
              width: 31%;
            }
          }
        }
        .selected-box-item {
          background-color: rgba(240, 240, 240, 0.9);
        }
        .result {
          width: 100%;
          height: 100%;
        }
      }
    }
  }
}
</style>
