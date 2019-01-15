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
          <label>
            <input class="checkbox" type="checkbox" id="prediction-show-button" v-model="show_target"
              v-on:change="onChangeTargetCheckBox">
            Target
          </label>
        </div>
      </div>
      <div id="image-container">
        <div id="image-wrapper" :style="getSize">
          <img :src="img"/>
  
          <div id="cls" v-if="isTaskClassification"
            :style="getClassificationStyle(result)">
          </div>

          <div id="box" v-else-if="isTaskDetection"
            :class="{'selected-box': index === hoverBox}"
            @mouseenter="hoverBox=index"
            @mouseleave="hoverBox=null"
            :style="getBoxStyle(box)" v-for="(box, index) in result">
            <div id="box-label" :style="getBoxLabelColor(box.class)">&nbsp&nbsp{{box.name}}</div>
          </div>

          <div id="seg" v-if="isTaskSegmentation">
            <canvas id="canvas-modal"/>
          </div>

        </div>
      </div>
    </div>
    <div id="result">
      <div class="header">
        <span>No.</span>
        <span>Score</span>
        <span>Name</span>
      </div>
      <div id="result-container">
        <div id="cls-result" class="result" v-if="isTaskClassification">
          <div v-for="(item, index) in getClassificationTop3">
            <span>{{ index + 1 }}</span>
            <span>{{ item.index }}</span>
            <span>{{ item.score }}%</span>
          </div>
          <div v-if="!result">
            <span></span>
            <span>No prediction</span>
            <span></span>
          </div>
        </div>
        <div id="box-result" class="result" v-else-if="isTaskDetection">
          <div v-for="(r, index) in prediction"
            @mouseenter="hoverBox=index + target.length*((show_target)? 1:0)"
            @mouseleave="hoverBox=null"
            :class="{'selected-box-item': index + target.length * ((show_target)? 1:0) === hoverBox}">
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
          <div v-for="item in prediction_of_segmentation">
            <span>{{ item }}</span>
          </div>
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

export default {
  name: 'ModalImage',
  mounted: function () {
    this.$refs.container.focus()
    if (this.isTaskSegmentation) {
      let a = this.result
    }
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
      show_image: false, // Show image or not.
      show_target: false, // Show target or not.
      show_prediction: true // Show prediction result or not.
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
    target: function () {
      const index = this.modal_index
      const dataset = this.dataset
      const model = this.model
      const target = dataset.getValidTarget(index)
      // 'This is not used but needed for reacting to the change of checkbox'
      const _ = this.show_target
      if (!dataset || !model) return
      if (this.isTaskSegmentation) {
        this.loadSegmentationTargetArray({
          name: target.name,
          size: [
            parseInt(model.hyper_parameters.imsize_w),
            parseInt(model.hyper_parameters.imsize_h)],
          callback: (response) => {
            const item = response.data
            this.getSegmentationStyle(item)
          }
        })
        return
      }
      return target
    },
    prediction: function () {
      const model = this.model
      // 'This is not used but needed for reacting to the change of checkbox'
      const _ = this.show_prediction
      if (model) {
        const result = model.getValidResult(this.modal_index)
        if (result) {
          if (this.isTaskSegmentation) {
            this.getSegmentationStyle(result)
            return {
              recall: result.recall,
              precision: result.precision,
            }
          }
          return result
        }
      }
    },
    prediction_of_segmentation: function () {
      const index = this.modal_index
      const target = this.dataset.getValidTarget(index)
      const pred = this.prediction
      const recall = pred.recall
      const precision = pred.precision
      this.getSegmentationStyle(target)
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
    getSize: function () {
      const size = this.size
      let w = size[0]
      let h = size[1]
      let parentW = (this.vw(60) - 40) * 0.65
      let parentH = this.vh(60) - 40 - 40 - 10
      if (parentW - w < parentH - h) {
        w = parentW
        h = parentW * h / w
      } else {
        w = parentH * w / h
        h = parentH
      }
      return {
        width: w + 'px',
        height: h + 'px',
      }
    },
    getClassificationTop3: function () {
      const map = this.class_map
      if (!this.prediction) return
      const score = this.prediction.score.map((s, index) => {
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
      return top5.map(d => { return {index: d.index, score: d.score.toFixed(2)} })
    },
    result: function () {
      if (this.isTaskDetection) {
        let result = []
        if (this.show_target) {
          result = [...this.target]
        }
        if (this.show_prediction) {
          if (this.prediction) {
            result = [...result, ...this.prediction]
          } else {
            result = [...result]
          }
        }
        return result
      } else {
        if (this.show_target) {
          return this.target
        } else if (this.show_prediction) {
          return this.prediction
        }
      }
      return []
    }
  },
  watch: {
    modal_index: function () {
      if (this.isTaskSegmentation) {
        this.result
      }
    },
  },
  methods: {
    ...mapMutations(['setImageModalData']),
    ...mapActions(['loadSegmentationTargetArray']),
    onChangePredictionCheckBox: function (e) {
      this.show_prediction = e.target.checked
      this.show_target = (!this.show_prediction || this.isTaskDetection) && this.show_target
      const a = this.result
    },
    onChangeTargetCheckBox: function (e) {
      this.show_target = e.target.checked
      this.show_prediction = (!this.show_target || this.isTaskDetection) && this.show_prediction
      const a = this.result
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
    getClassificationStyle: function (cls) {
      if (!cls) {
        return {}
      }
      if (cls.hasOwnProperty('score') && cls.hasOwnProperty('class')) {
        const class_id = cls.class
        return {
          border: 'solid 3px' + getTagColor(class_id) + 'bb'
        }
      } else {
        const class_id = cls
        return {
          border: 'solid 3px' + getTagColor(class_id) + 'bb'
        }
      }
    },
    getBoxStyle: function (box) {
      if (!box) return
      const class_id = box.class
      const x1 = (box.box[0] - box.box[2] / 2) * 100
      const y1 = (box.box[1] - box.box[3] / 2) * 100
      return {
        top: y1 + '%',
        left: x1 + '%',
        width: box.box[2] * 100 + '%',
        height: box.box[3] * 100 + '%',
        border: 'solid 2.5px' + getTagColor(class_id) + 'bb'
      }
    },
    getBoxLabelColor: function (class_id) {
      return {
        'background-color': getTagColor(class_id) + 'bb'
      }
    },
    getSegmentationStyle: function (item) {
      if (!item) return
      if (!item) {
        // Clear canvas
        var canvas = document.getElementById('canvas-modal')
        var cxt = canvas.getContext('bitmaprenderer')
        var offCanvas = new OffscreenCanvas(canvas.width, canvas.height)
        var offCxt = offCanvas.getContext('2d')
        offCxt.clearRect(0, 0, canvas.width, canvas.height)
        cxt.transferFromImageBitmap(offCanvas.transferToImageBitmap())
        return
      }
      this.$worker.run(render_segmentation, [item]).then((ret) => {
        var canvas = document.getElementById('canvas-modal')
        var cxt = canvas.getContext('bitmaprenderer')
        cxt.transferFromImageBitmap(ret)
      })
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
      #image-wrapper {
        width: 100%;
        height: 100%;
        position: relative;
        img {
          width: 100%;
          height: 100%;
        }
        #cls {
          top: 0;
          left: 0;
          position: absolute;
          height: 100%;
          width: 100%;
        }
        #box {
          position: absolute;
          height: 100%;
          width: 100%;
          #box-label {
            display: flex;
            min-width: 100%;
            height: calc(20px - 2.5px);
            position: relative;
            background-color: white;
            color: white;
            font-size: 0.8rem;
          }
        }
        #seg {
          top: 0;
          left: 0;
          position: absolute;
          height: 100%;
          width: 100%;
          canvas {
            height: 100%;
            width: 100%;
          }
        }
        .selected-box {
          background-color: rgba(255, 255, 255, 0.4);
        }
      }
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
