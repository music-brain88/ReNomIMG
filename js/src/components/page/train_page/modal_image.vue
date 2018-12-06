<template>
  <div id="image-modal" ref="container"
    v-on:keyup.right="nextPage"
    v-on:keyup.left="prevPage"
    tabindex="0">
    <div id="image-result">
      <div class="header">
        <span>Prediction Result</span>
        <span>{{modal_index}} / {{length}}</span>
      </div>
      <div id="image-wrapper" :style="getSize">
        <img :src="img"/>

        <div id="cls" v-if="isTaskClassification"
          :style="getClassificationStyle(prediction)">
        </div>

        <div id="box" v-else-if="isTaskDetection"
          :class="{'selected-box': index === hoverBox}"
          @mouseenter="hoverBox=index"
          @mouseleave="hoverBox=null"
          :style="getBoxStyle(box)" v-for="(box, index) in prediction">
        </div>

        <div id="seg" v-if="isTaskSegmentation">
          <canvas id="canvas-modal"/>
        </div>

      </div>
    </div>
    <div id="result">
      <div class="header"></div>
      <div id="result-container">

        <div id="cls-result" class="result" v-if="isTaskClassification">
        </div>

        <div id="box-result" class="result" v-else-if="isTaskDetection">
          <div v-for="(r, index) in prediction"
            @mouseenter="hoverBox=index"
            @mouseleave="hoverBox=null"
            :class="{'selected-box-item': index === hoverBox}">
            <span>{{index}}</span>
            <span>{{r.score.toFixed(2)}}</span>
            <span>{{r.name}}-{{r.class}}</span>
          </div>
          <div v-if="prediction.length === 0">
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
import { mapGetters, mapState, mapMutations } from 'vuex'
import { getTagColor, render_segmentation, setup_image_list } from '@/utils.js'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ModalImage',
  mounted: function () {
    this.$refs.container.focus()
    if (this.isTaskSegmentation) {
      this.getSegmentationStyle(this.prediction)
    }
  },
  data: function () {
    return {
      hoverBox: null
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
    prediction: function () {
      const model = this.model
      if (model) {
        const result = model.getValidResult(this.modal_index)
        if (result) {
          return result
        }
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
      let parentW = (this.vw(60) - 20) * 0.65
      let parentH = this.vh(60) - 20 - 40 - 10
      if (parentW / parentH < w / h) {
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
    }
  },
  watch: {
    modal_index: function () {
      if (this.isTaskSegmentation) {
        this.getSegmentationStyle(this.prediction)
      }
    }
  },
  methods: {
    ...mapMutations(['setImageModalData']),
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
        border: 'solid 5px' + getTagColor(class_id) + 'bb'
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
  #image-result {
    width: 65%;
    height: 100%;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    #image-wrapper {
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
  #result {
    width: 35%;
    height: 100%;
    margin-left: 10px;
    #result-container {
      width: 100%;
      height: calc(100% - 40px - 10px);
      #box-result div {
        display: flex;
        width: 100%;
        height: 9%;
        border-bottom: solid 1px lightgray;
        span {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: space-around;
          width: 33.3%;
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
    height: 40px;
    width: 100%;
    background-color: $header-background-color;
    margin-bottom: 10px;
    color: white;
    display: flex;
    align-items: center;
    padding-left: 10px;
    padding-right: 10px;
    justify-content: space-between;
  }
}
</style>
