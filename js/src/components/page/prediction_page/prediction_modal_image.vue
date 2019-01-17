<template>
  <div id="image-modal" ref="container"
    v-on:keyup.right="nextPage"
    v-on:keyup.left="prevPage"
    tabindex="0">
    <div id="image-result">
      <div class="header">
        <span style="width: 50%">Prediction Result &nbsp;&nbsp; {{modal_index+1}} / {{length}}</span>
        <div id="checkbox-area">
          <label>
            <input class="checkbox" type="checkbox" id="prediction-show-button" v-model="show_prediction"
              v-on:change="onChangePredictionCheckBox">
            Prediction
          </label>
        </div>
      </div>
      <div id="image-container" ref="imageContainer">
        <image-canvas
          :show-target="show_target"
          :show-predict="show_prediction"
          :show-image="show_image"
          :img="img"
          :width="size[0]"
          :height="size[1]"
          :maxWidth="canvas_width"
          :maxHeight="canvas_height"
          :model="getDeployedModel"
          :result="getResult()"
        />
      </div>
    </div>
    <div id="result" v-if="isTaskDetection">
      <div class="header">
        <span>No.</span>
        <span>Name</span>
        <span>Score</span>
      </div>
      <div id="result-container">
        <div id="cls-result" class="result" v-if="isTaskClassification">
        </div>
        <div id="box-result" class="result" v-else-if="isTaskDetection">
          <div v-for="(r, index) in prediction"
            @mouseenter="hoverBox=index? 1:0"
            @mouseleave="hoverBox=null">
            <span>{{index}}</span>
            <span>{{r.score.toFixed(2)}}</span>
            <span>{{r.name}}</span>
          </div>
          <div v-if="!prediction || prediction.length === 0">
            <span></span>
            <span>No Prediction</span>
            <span></span>
          </div>
        </div>
        <div id="seg-result" class="result" v-else-if="isTaskSegmentation">
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { TASK_ID } from '@/const.js'
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import { getTagColor, render_segmentation, setup_image_list } from '@/utils.js'
import ComponentFrame from '@/components/common/component_frame.vue'
import ImageCanvas from '@/components/page/train_page/image.vue'

export default {
  name: 'PredictionModalImage',
  components: {
    'image-canvas': ImageCanvas,
  },
  mounted: function () {
    this.$refs.container.focus()
    const el = this.$refs.imageContainer
    this.canvas_width = el.clientWidth
    this.canvas_height = el.clientHeight
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
      'getDeployedModel',
      'getCurrentTask',
      'isTaskClassification',
      'isTaskDetection',
      'isTaskSegmentation',
    ]),
    model: function () {
      const model = this.getDeployedModel
      if (model) {
        return model
      }
      return null
    },
    prediction: function () {
      const index = this.modal_index
      return this.prediction_dataset.prediction[index]
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
    prediction_dataset: function () {
      const model = this.model
      if (model) {
        return model.prediction_result
      }
    },
    class_map: function () {
      const map = this.dataset.class_map
      return map
    },
    img: function () {
      const index = this.modal_index
      return this.prediction_dataset.img[index]
    },
    size: function () {
      const index = this.modal_index
      return this.prediction_dataset.size[index]
    },
    length: function () {
      return this.prediction_dataset.img.length
    },
  },
  methods: {
    ...mapMutations(['setImageModalData']),
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
      const pred = this.prediction
      return {
        index: index,
        target: undefined,
        predict: pred
      }
    },
  },
}
</script>

<style lang='scss'>

#image-modal {
  width: 100%;
  height: 100%;
  display: flex;
  padding: 10px;
  #image-result {
    width: 65%;
    height: 100%;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    #image-container {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      height: calc(100% - 40px - 10px);
    }
  }
  #result {
    width: 35%;
    height: 100%;
    margin-left: 10px;
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
    }
    .result {
      width: 100%;
      height: 100%;
    }
  }
  .header {
    height: 32px;
    width: 100%;
    background-color: $header-background-color;
    margin-bottom: 10px;
    color: white;
    display: flex;
    align-items: center;
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
    #checkbox-area {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      height: 100%;
      width: 50%;
      label { 
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: $component-header-font-family;
        font-size: 90%;
        margin-right: 10px;
      }
      input {
        display: none;
        -webkit-appearance: none;
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
}
</style>
