<template>
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <!--Header Contents-->
    <template slot="header-slot">
      Prediction Result
      <div
        id="valid-prediction-button-area"
        tabindex="0"
        @keyup.right="nextPage"
        @keyup.left="prevPage"
      >
        <!--
          Only if Segmentation, "show image toggle" will be shown.
        -->
        <!--
          'Show prediction result' Button and 'show target' Button.
          - In classification, prediction or target can be on.
          - In detection, both prediction and target can be on.
          - In segmentation, prediction or target can be on.
        -->

        <rnc-button-switch
          v-if="isTaskSegmentation"
          :checked="show_image"
          :label="'Image'"
          :disabled="!isTaskSegmentation"
          @change="show_image = !show_image"
        />
        <rnc-button-switch
          :checked="show_prediction"
          :label="'Prediction'"
          @change="onChangePredictionCheckBox"
        />
        <rnc-button-switch
          :checked="show_target"
          :label="'Target'"
          @change="onChangeTargetCheckBox"
        />
      </div>
    </template>
    <!--Header Contents-->

    <template slot="content-slot">
      <!--Pager Settings-->
      <div class="pager">
        <rnc-pager
          :page-max="pageMax"
          @set-page="setImagePageOfValid"
        />
      </div>
      <!-- :on-set-page="setImagePageOfValid" -->

      <!--Pager Settings-->

      <!--Image list-->
      <div
        id="img-container"
        ref="container"
      >
        <rnc-labeled-image
          v-for="(item, key) in getValidImages"
          :key="key"
          :callback="() => {showImageModal(item)}"
          :show-target="show_target"
          :show-predict="show_prediction"
          :show-image="show_image"
          :width="item.size[0]"
          :height="item.size[1]"
          :max-height="image_cell_height"
          :img="item.img"
          :result="getResult(item)"
          :model="getSelectedModel"
          :dataset="dataset"
        />
      </div>
      <!--Image list-->
    </template>
  </rnc-title-frame>
</template>

<script>
import { mapGetters, mapState, mapMutations } from 'vuex'
import { setup_image_list } from '../../../utils.js'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncLabeledImage from './../../Molecules/rnc-labeled-image/rnc-labeled-image.vue'
import RncPager from './../../Molecules/rnc-pager/rnc-pager.vue'
import RncButtonSwitch from './../../Atoms/rnc-button-switch/rnc-button-switch.vue'

// const RESERVED = '-1'  // TODO: 「'RESERVED' is assigned a value but never used」のためコメントアウト
// const CREATED = '-2'

export default {
  name: 'RncTrainPanelPredictionResult',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-labeled-image': RncLabeledImage,
    'rnc-pager': RncPager,
    'rnc-button-switch': RncButtonSwitch
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 7,
    }
  },
  data: function () {
    return {
      // The state of the checkbox which represents weather image is shown.
      show_image: true,
      // The state of the checkbox which represents weather target is shown.
      show_target: false,
      // The state of the checkbox which represents weather prediction is shown.
      show_prediction: true,
      image_cell_height: 0
    }
  },
  computed: {
    ...mapState([
      'datasets',
    ]),
    ...mapGetters([
      'getSelectedModel',
      'getCurrentTask',
      'getImagePageOfValid', // This will return current page number of image list.
      'isTaskClassification',
      'isTaskDetection',
      'isTaskSegmentation',
    ]),
    dataset: function () {
      const model = this.getSelectedModel
      if (model) return this.datasets.find(d => d.id === model.dataset_id)
    },
    showImage: function () {
      // If the task is segmentation, image show button will be appeared.
      return this.show_image || !this.isTaskSegmentation
    },
    getValidImages: function () {
      /**
        If the dataset obj has no page, new image list will be created and
        saved to 'dataset.page', otherwise 'dataset.page' will be returned.

        Returns : Image list of current page.
      */
      const dataset = this.dataset
      if (!dataset) return
      console.log('***dataset***')
      console.log(dataset)

      if (dataset.page.length === 0) {
        // Setup image page if it has not been set.
        this.$nextTick(() => this.setUpValidImages())
      }

      // Clip page number.
      let current_page = this.getImagePageOfValid
      const max_page_num = dataset.page.length - 1
      const page_num = Math.max(Math.min(current_page, max_page_num), 0)
      this.setImagePageOfValid(page_num)
      current_page = this.getImagePageOfValid
      console.log('【dataset.page】')
      console.log(JSON.stringify(dataset.page[current_page]))
      return dataset.page[current_page]
    },
    pageMax: function () {
      if (this.dataset && this.dataset.page.length > 0) {
        return this.dataset.page.length
      }
    },
  },
  mounted: function () {
    const el = this.$refs.container
    if (el) {
      this.image_cell_height = el.clientHeight / 3
    }
  },
  beforeUpdate: function () {
    /**
      If the task is segmentation, drawing function will be called in
      each update.
    */
  },
  methods: {
    ...mapMutations([
      'setImagePageOfValid', // Set current page number.
      'showModal',
      'setImageModalData' // This will set index of image for show in modal.
    ]),
    onChangePredictionCheckBox: function (e) {
      this.show_prediction = e.target.checked
      this.show_target = (!this.show_prediction || this.isTaskDetection) && this.show_target
      // const _ = this.result  // TODO: 「'_' is assigned a value but never used」のためコメントアウト
    },
    onChangeTargetCheckBox: function (e) {
      this.show_target = e.target.checked
      this.show_prediction = (!this.show_target || this.isTaskDetection) && this.show_prediction
      // const a = this.result  // TODO: 「'a' is assigned a value but never used」のためコメントアウト
    },
    showImageModal: function (item) {
      /**
        The image modal will appear.
      */
      this.setImageModalData(item.index)
      this.showModal({ 'show_image': true })
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
    setUpValidImages: function () {
      /**
        This function will create page list of image like following example.
        [
          [img1.jpg, img2.jpg, img3.jpg...], // Page1
          [img11.jpg, img12.jpg, img13.jpg...], // Page2
          ...
        ]
      */
      const parent_div = this.$refs.container
      if (parent_div === undefined) return
      const parent_height = parent_div.clientHeight
      const parent_width = parent_div.clientWidth
      const child_margin = Math.min(this.vh(0.25), this.vw(0.25))

      const dataset = this.dataset
      if (!dataset) return
      console.log("【dataset.valid_data】")
      console.log(dataset.valid_data)
      // Using vue-worker here.
      // See https://github.com/israelss/vue-worker
      this.$worker.run(setup_image_list,
        [dataset.valid_data, parent_width, parent_height, child_margin])
        .then((ret) => {
          dataset.page = ret
        })
    },
    getResult: function (item) {
      const index = item.index
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
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';


.rnc-title-frame {
  .component-header {
    #valid-prediction-button-area {
      display: flex;
      align-items: center;
      justify-content: flex-end;
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
        margin-right: 10px;
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
  }

  .frame-content {
    .pager {
      padding-right: $padding-small;
      height: 5%;
      #pager {
        justify-content: flex-end;
      }
    }

    #img-container{
      width: 100%;
      height: 95%;
      display: flex;
      flex-wrap: wrap;
      div:hover {
        cursor: pointer;
      }
    }
  }
}
</style>
