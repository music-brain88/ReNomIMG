<template>
  <component-frame
    :width-weight="8"
    :height-weight="9">

    <!--Header Contents-------------->
    <template slot="header-slot">
      Prediction Result
      <div
        id="prediction-button-area"
        tabindex="0"
        @keyup.right="nextPage"
        @keyup.left="prevPage">
        <!--Check Box. Switch of showing image and predicted box.------>
        <label>
          <input
            id="prediction-show-button"
            v-model="show_image"
            :disabled="!isTaskSegmentation"
            class="checkbox"
            type="checkbox">
          Image
        </label>

        <label>
          <input
            id="prediction-show-button"
            v-model="show_prediction"
            class="checkbox"
            type="checkbox">
          Prediction
        </label>
      </div>

    </template>
    <div id="prediction-area">
      <pager
        :page-max="page.length"
        :on-set-page="setImagePageOfPrediction"
      />
      <div
        id="img-container"
        ref="container">
        <div
          v-if="showResult"
          id="img-list">
          <image-frame
            v-for="(item, index) in getImages"
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
          />
        </div>
        <div
          v-else
          id="progress-animation">
          <div
            v-if="getImages.length===0"
            id="no-prediction">
            No prediction
          </div>
          <div
            v-else
            id="predicting">
            <div class="lds-roller">
              <div/><div/><div/>
              <div/><div/><div/>
              <div/><div/>
            </div>
            <span>{{ pretidtionProgress }}</span>
          </div>
        </div>
      </div>
    </div>
  </component-frame>
</template>

<script>
import { TASK_ID } from '@/const.js'
import { getTagColor, render_segmentation, setup_image_list } from '@/utils.js'
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ImageCanvas from '@/components/page/train_page/image.vue'
import Pager from '@/components/page/train_page/pager.vue'

export default {
  name: 'ComponentPrediction',
  components: {
    'component-frame': ComponentFrame,
    'image-frame': ImageCanvas,
    'pager': Pager
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
      } else {
      }
    },
    showImage: function () {
      return this.show_image || !this.isTaskSegmentation
    },
    getImages: function () {
      const model = this.model
      if (model) {
        let current_page = this.getImagePageOfPrediction
        const dataset = model.prediction_result
        if (!dataset) {
          this.page = []
          return []
        }
        if (this.page.length === 0) {
          // Setup image page if it has not been set.
          this.setUpImages()
        }

        // Clip page number.
        const max_page_num = this.page.length - 1
        const page_num = Math.max(Math.min(current_page, max_page_num), 0)
        this.setImagePageOfPrediction(page_num)
        current_page = this.getImagePageOfPrediction
        return this.page[current_page]
      }
      this.page = []
      return []
    },
    showResult () {
      const model = this.model
      if (!model) return false
      return model.isStopped()
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
  methods: {
    ...mapMutations(['setImagePageOfPrediction', 'showModal', 'setImageModalData']),
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
  }
}
</script>

<style lang='scss'>
#prediction-button-area {
  display: flex;
  align-items: center;
  justify-content: space-around;
  height: 100%;
  width: 30%;
  input {
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
