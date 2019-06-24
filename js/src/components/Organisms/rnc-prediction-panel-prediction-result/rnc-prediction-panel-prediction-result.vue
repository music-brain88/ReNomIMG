<template>
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <!--Header Contents-->
    <template slot="header-slot">
      Prediction Result
      <div
        id="prediction-button-area"
        tabindex="0"
        @keyup.right="nextPage"
        @keyup.left="prevPage"
      >
        <!--Check Box. Switch of showing image and predicted box-->
        <rnc-button-switch
          :checked="show_image"
          :disabled="!isTaskSegmentation"
          :label="'Image'"
          class="margin-right"
          @change="show_image = $event.target.checked"
        />
        <rnc-button-switch
          :checked="show_prediction"
          :label="'Prediction'"
          class="margin-right"
          @change="show_prediction = $event.target.checked"
        />

        <input
          :disabled="!(!isPredicting && showResult)"
          type="button"
          value="Download"
          @click="onDownload"
        >
      </div>
    </template>

    <template slot="content-slot">
      <div id="prediction-area">
        <div class="pager">
          <rnc-pager
            :page-max="page.length"
            @set-page="setImagePageOfPrediction"
          />
        </div>

        <div
          id="img-container"
          ref="container"
        >
          <div
            v-if="showResult"
            id="img-list"
          >
            <rnc-labeled-image
              v-for="(item, index) in getImages"
              :key="index"
              :callback="() => {showImageModal(item)}"
              :show-target="show_target"
              :show-predict="show_prediction"
              :show-image="show_image"
              :width="item.size[0]"
              :height="item.size[1]"
              :max-height="$refs.container.clientHeight/3"
              :img="item.img"
              :result="getResult(item)"
              :model="model"
              ::dataset="dataset"
            />
          </div>
          <div
            v-else
            id="progress-animation"
          >
            <div
              v-if="!isPredicting"
              id="no-prediction"
            >
              No prediction
            </div>
            <div
              v-else
              id="predicting"
            >
              <div class="lds-roller">
                <div />
                <div />
                <div />
                <div />
                <div />
                <div />
                <div />
                <div />
              </div>
              <span>
                {{ pretidtionProgress }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </template>
  </rnc-title-frame>
</template>

<script>
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import { setup_image_list } from '../../../utils.js'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncLabeledImage from './../../Molecules/rnc-labeled-image/rnc-labeled-image.vue'
import RncPager from './../../Molecules/rnc-pager/rnc-pager.vue'
import RncButtonSwitch from './../../Atoms/rnc-button-switch/rnc-button-switch.vue'

// const RESERVED = '-1'  // TODO:「'RESERVED' is assigned a value but never used 」のため取り急ぎコメントアウト
// const CREATED = '-2'

export default {
  name: 'RncPredictionPanelPredictionResult',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-labeled-image': RncLabeledImage,
    'rnc-pager': RncPager,
    'rnc-button-switch': RncButtonSwitch
  },
  props: {
    widthWeight: {
      type: Number,
      default: 8
    },
    heightWeight: {
      type: Number,
      default: 9
    }
  },
  data: function () {
    return {
      show_image: true,
      show_target: false,
      show_prediction: true,
      page: []
    }
  },
  computed: {
    ...mapState([
      'datasets',
      'modal_image',
      'modal_prediction',
      'modal_target'
    ]),
    ...mapGetters([
      'getCurrentTask',
      'getImagePageOfPrediction',
      'getDeployedModel',
      'isTaskClassification',
      'isTaskDetection',
      'isTaskSegmentation',
    ]),
    model: function () {
      const model = this.getDeployedModel
      if (model) {
        return model
      }
    },
    showImage: function () {
      return this.show_image || !this.isTaskSegmentation
    },
    getImages: function () {
      const model = this.model
      if (model) {
        const dataset = model.prediction_result
        if (!dataset) {
          return []
        }
        if (this.page.length === 0) {
          // Setup image page if it has not been set.
          this.$nextTick(() => this.setUpImages())
        }

        // Clip page number.
        let current_page = this.getImagePageOfPrediction
        const max_page_num = this.page.length - 1
        const page_num = Math.max(Math.min(current_page, max_page_num), 0)
        this.setImagePageOfPrediction(page_num)
        current_page = this.getImagePageOfPrediction
        return this.page[current_page]
      }
      return []
    },
    isPredicting: function () {
      const model = this.model
      if (!model) return false
      return model.isPredicting()
    },
    showResult () {
      const images = this.getImages
      const model = this.model
      if (!model || !images) return false
      return model.isStopped() && (images.length > 0)
    },
    pretidtionProgress () {
      const model = this.model
      if (model) {
        const total = model.total_prediction_batch
        const nth = model.nth_prediction_batch
        if (total === 0) {
          return '0.00 %'
        }
        return (nth / total * 100).toFixed(2) + ' %'
      }
      return '-'
    }
  },
  watch: {
    getImages: function () {
      if (this.getImages === []) {
        this.page = []
      }
    }
  },
  methods: {
    ...mapMutations(['setImagePageOfPrediction', 'showModal', 'setImageModalData']),
    ...mapActions(['downloadPredictionResult']),
    showImageModal: function (item) {
      this.setImageModalData(item.index)
      this.showModal({ 'show_prediction_image': true })
    },
    vh: function (v) {
      var h = Math.max(document.documentElement.clientHeight,
        window.innerHeight || 0)
      return (v * h) / 100
    },
    vw: function (v) {
      var w = Math.max(document.documentElement.clientWidth,
        window.innerWidth || 0)
      return (v * w) / 100
    },
    setUpImages: function () {
      const parent_div = this.$refs.container
      if (!parent_div) return
      const parent_height = parent_div.clientHeight
      const parent_width = parent_div.clientWidth
      const child_margin = Math.min(this.vh(0.25), this.vw(0.25))

      const model = this.model
      if (!model) return

      const dataset = model.prediction_result
      if (!dataset) return

      // Using vue-worker here.
      // See https://github.com/israelss/vue-worker
      this.$worker.run(setup_image_list, [dataset, parent_width, parent_height, child_margin])
        .then((ret) => {
          this.page = ret
        })
    },
    getResult: function (item) {
      const index = item.index
      const model = this.model
      if (!model) return
      const pred = model.prediction_result.prediction[index]
      return {
        index: index,
        target: undefined,
        predict: pred
      }
    },
    onDownload: function () {
      const page = this.page
      if (!page) return
      // Save and download csv here.
      if (this.model) {
        this.downloadPredictionResult(this.model)
      }
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.margin-right {
  margin-right: 10px;
}

.pager {
  padding-right: $padding-small;
  #pager {
    justify-content: flex-end;
  }
}

#prediction-button-area {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  width: 36%;
  input[type="checkbox"] {
    display: none;
    -webkit-appearance: none;
  }
  label {
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: $component-header-font-family;
    font-size: 90%;
  }
  input[type="button"] {
    height: 100%;
    white-space: pre-line;
    word-break: break-all;
  }
  input[type="checkbox"] {
    content: "";
    display: block;
    height: 12px;
    width: 12px;
    border: 1px solid white;
    border-radius: 6px;
  }
  input[type="checkbox"]:checked {
    content: "";
    display: block;
    border: 1px solid white;
    background-color: white;
  }
  input[type="checkbox"]:disabled {
    content: "";
    display: block;
    border: 1px solid gray;
    background-color: gray;
  }
  input[type="checkbox"]:focus {
      outline:none;
  }
}

#prediction-area {
  width: 100%;
  height: 100%;
  #img-container{
    width: 100%;
    height: 95%;
    display: flex;
    flex-wrap: wrap;
    #img-list {
      width: 100%;
      height: 100%;
      display: flex;
      flex-wrap: wrap;
    }
    #progress-animation {
      width: 100%;
      height: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      #no-prediction {
        width: 100%;
        text-align: center;
      }
      #predicting {
        display: flex;
        flex-direction: column;
        text-align: center;
        .lds-roller {
          display: inline-block;
          position: relative;
          width: 64px;
          height: 64px;
        }
        .lds-roller div {
          animation: lds-roller 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
          transform-origin: 32px 32px;
        }
        .lds-roller div:after {
          content: " ";
          display: block;
          position: absolute;
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #aaa;
          margin: -3px 0 0 -3px;
        }
        .lds-roller div:nth-child(1) {
          animation-delay: -0.036s;
        }
        .lds-roller div:nth-child(1):after {
          top: 50px;
          left: 50px;
        }
        .lds-roller div:nth-child(2) {
          animation-delay: -0.072s;
        }
        .lds-roller div:nth-child(2):after {
          top: 54px;
          left: 45px;
        }
        .lds-roller div:nth-child(3) {
          animation-delay: -0.108s;
        }
        .lds-roller div:nth-child(3):after {
          top: 57px;
          left: 39px;
        }
        .lds-roller div:nth-child(4) {
          animation-delay: -0.144s;
        }
        .lds-roller div:nth-child(4):after {
          top: 58px;
          left: 32px;
        }
        .lds-roller div:nth-child(5) {
          animation-delay: -0.18s;
        }
        .lds-roller div:nth-child(5):after {
          top: 57px;
          left: 25px;
        }
        .lds-roller div:nth-child(6) {
          animation-delay: -0.216s;
        }
        .lds-roller div:nth-child(6):after {
          top: 54px;
          left: 19px;
        }
        .lds-roller div:nth-child(7) {
          animation-delay: -0.252s;
        }
        .lds-roller div:nth-child(7):after {
          top: 50px;
          left: 14px;
        }
        .lds-roller div:nth-child(8) {
          animation-delay: -0.288s;
        }
        .lds-roller div:nth-child(8):after {
          top: 45px;
          left: 10px;
        }
        @keyframes lds-roller {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }
      }
    }
  }
}
</style>
