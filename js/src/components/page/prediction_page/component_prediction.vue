<template>
  <component-frame :width-weight="8" :height-weight="9">

    <!--Header Contents-------------->
    <template slot="header-slot">
      Prediction Result
      <div id="prediction-button-area" tabindex="0"
        v-on:keyup.right="nextPage" v-on:keyup.left="prevPage">
        <!--Check Box. Switch of showing image and predicted box.------>
        <label>
          <input class="checkbox" type="checkbox"
            id="prediction-show-button" v-model="show_image" :disabled="!isTaskSegmentation">
          Image
        </label>

        <label>
          <input class="checkbox" type="checkbox"
            id="prediction-show-button" v-model="show_prediction">
          Prediction
        </label>
      </div>

    </template>
    <div id="prediction-area">
      <pager
        :page-max="page.length"
        :onSetPage="setImagePageOfPrediction"
      />
      <div id="img-container" ref="container">
        <image-frame v-for="(item, index) in getImages"
          :callback="() => {showImageModal(item)}"
          :show-target="show_target"
          :show-predict="show_prediction"
          :show-image="show_image"
          :width="item.size[0]" :height="item.size[1]"
          :maxHeight="$refs.container.clientHeight/3"
          :img="item.img"
          :result="getResult(item)"
          :model="model"
        />
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
      return []
    },
  },
  methods: {
    ...mapMutations(['setImagePageOfPrediction', 'showModal', 'setImageModalData']),
    showImageModal: function (item) {
      this.setImageModalData(item.index)
      this.showModal({'show_image': true})
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
      const pred = model.getValidResult(index)
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
  }
}
</style>
