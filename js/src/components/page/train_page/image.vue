<template>
  <div id="image-frame" v-on:click="onImageClick()"
    :style="modifiedSize" ref="wrapper">
    <canvas id="seg" v-if="isTaskSegmentation" ref="canvas"/>
    <div id="box" v-if="isTaskDetection"
      @mouseenter="boxEnter(b)"
      @mouseleave="boxLeave(b)"
      :style="styleBox(b)" v-for="b in box">
      <div id="box-label" :style="styleBoxLabel(b)">
        {{b.name}}
      </div>
    </div>
    <div id="cls" v-if="isTaskClassification" :style="styleCls()">
      <div id="cls-label" :style="styleClsLabel()">
        {{ cls }}
      </div>
    </div>
    <img :src="img" :style="modifiedSize" v-if="showImage"/>
  </div>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex'
import { TASK_ID } from '@/const.js'
import { getTagColor, render_segmentation } from '@/utils.js'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ImageCanvas',
  components: {
    'component-frame': ComponentFrame
  },
  props: {
    width: {
      type: Number,
      default: 0
    },
    height: {
      type: Number,
      default: 0
    },
    maxWidth: {
      type: Number,
      default: 0
    },
    maxHeight: {
      type: Number,
      default: 0
    },
    img: {
      type: String,
      default: ''
    },
    model: {
      type: Object,
      default: undefined
    },
    callback: {
      default: (result) => {}
    },
    boxEnterCallback: {
      default: (result) => {}
    },
    boxLeaveCallback: {
      default: (result) => {}
    },
    // Followings are Object of predicted and target data.
    result: {
      index: -1,
      target: undefined,
      predict: undefined,
    },
    showPredict: {
      default: false
    },
    showTarget: {
      default: true
    },
    showImage: {
      default: true
    }
  },
  beforeUpdate: function () {
    /**
      If the task is segmentation, drawing function will be called in
      each update.
    */
    this.$nextTick(function () {
      if (this.isTaskSegmentation) {
        this.drawSeg()
      }
    })
  },
  computed: {
    ...mapState([
      'datasets',
    ]),
    ...mapGetters([
      'getCurrentTask',
      'getImagePageOfValid', // This will return current page number of image list.
      'isTaskClassification',
      'isTaskDetection',
      'isTaskSegmentation',
    ]),
    dataset: function () {
      const model = this.model
      if (model) return this.datasets.find(d => d.id === model.dataset_id)
    },
    modifiedSize: function () {
      let w, h
      if (this.maxWidth === 0) {
        if (this.maxHeight == 0) {
          w = this.width
          h = this.height
        } else {
          let r = this.maxHeight / this.height
          w = this.width * r
          h = this.height * r
        }
      } else if (this.maxHeight === 0) {
        if (this.maxWidth == 0) {
          // Never reach here
        } else {
          let r = this.maxWidth / this.width
          w = this.width * r
          h = this.height * r
        }
      } else {
        let wr = this.maxWidth / this.width
        let hr = this.maxHeight / this.height
        let r = (wr < hr) ? wr : hr
        w = this.width * r
        h = this.height * r
      }
      return {
        width: w + 'px',
        height: h + 'px',
      }
    },
    box: function () {
      if (!this.isTaskDetection) return
      let pred = this.result.predict
      let targ = this.result.target
      if (targ === undefined || !this.showTarget) {
        targ = []
      }
      if (pred === undefined || !this.showPredict) {
        pred = []
      }
      return pred.concat(targ)
    },
    cls: function () {
      const dataset = this.dataset
      if (!this.isTaskClassification || !dataset) return
      const class_map = dataset.class_map
      if (this.showTarget) {
        const targ = this.result.target
        if (!targ) return
        return class_map[targ]
      } else if (this.showPredict) {
        const pred = this.result.predict
        if (!pred) return
        return class_map[pred.class]
      }
    }
  },
  mounted: function () {
    let container = this.$refs.wrapper
    if (!container) return
    this.image_width = this.modifiedWidth
    this.image_height = this.modifiedHeight
  },
  methods: {
    ...mapActions([
      'loadSegmentationTargetArray' // Get segmentation target from server.
    ]),
    onImageClick: function () {
      if (this.callback) {
        this.callback(this.result)
      }
    },
    boxEnter: function (box) {
      if (this.boxEnterCallback) {
        this.boxEnterCallback()
      }
    },
    boxLeave: function () {
      if (this.boxLeaveCallback) {
        this.boxLeaveCallback(this.result)
      }
    },
    clamp: function (val, max, min) {
      return Math.max(Math.min(val, max), min)
    },
    styleCls: function () {
      let cls
      if (this.showTarget) {
        cls = this.result.target
      } else if (this.showPredict) {
        cls = this.result.predict.class
      }
      return {
        border: 'solid 2px' + getTagColor(cls) + 'bb'
      }
    },
    styleClsLabel: function () {
      let cls
      if (this.showTarget) {
        cls = this.result.target
      } else if (this.showPredict) {
        cls = this.result.predict.class
      }
      return {
        'background-color': getTagColor(cls) + 'bb'
      }
    },
    styleBox: function (box) {
      const class_id = box.class
      const x1 = (box.box[0] - box.box[2] / 2) * 100
      const y1 = (box.box[1] - box.box[3] / 2) * 100
      const w = box.box[2] * 100
      const h = box.box[3] * 100
      return {
        top: y1 + '%',
        left: x1 + '%',
        width: w + '%',
        height: h + '%',
        border: 'solid 2px' + getTagColor(class_id) + 'bb'
      }
    },
    styleBoxLabel: function (box) {
      const class_id = box.class
      return {
        'background-color': getTagColor(class_id) + 'bb'
      }
    },
    drawSeg: function () {
      let draw_item
      if (!this.showPredict && !this.showTarget) {
        var canvas = this.$refs.canvas
        if (!canvas) return
        var cxt = canvas.getContext('bitmaprenderer')
        var offCanvas = new OffscreenCanvas(canvas.width, canvas.height)
        var offCxt = offCanvas.getContext('2d')
        offCxt.clearRect(0, 0, canvas.width, canvas.height)
        cxt.transferFromImageBitmap(offCanvas.transferToImageBitmap())
      } else if (this.showPredict) {
        draw_item = this.result.predict
        if (draw_item === undefined) return
        this.$worker.run(render_segmentation, [draw_item]).then((ret) => {
          var canvas = this.$refs.canvas
          var cxt = canvas.getContext('bitmaprenderer')
          cxt.transferFromImageBitmap(ret)
        })
      } else if (this.showTarget) {
        const model = this.model
        draw_item = this.result.target
        if (!model || draw_item.name === undefined) return
        this.loadSegmentationTargetArray({
          name: draw_item.name,
          size: [
            parseInt(model.hyper_parameters.imsize_w),
            parseInt(model.hyper_parameters.imsize_h),
          ],
          callback: (response) => {
            this.$worker.run(render_segmentation, [response.data]).then((ret) => {
              console.log(ret)
              var canvas = this.$refs.canvas
              var cxt = canvas.getContext('bitmaprenderer')
              cxt.transferFromImageBitmap(ret)
            })
          }
        })
      }
    },
  }
}
</script>

<style lang='scss'>
#image-frame {
  display: flex;
  flex-grow: 1;
  flex-shrink: 1;
  overflow: hidden;
  position: relative;
  canvas {
    position: absolute;
    width: 100%;
    height: 100%;
  }
  #box {
    position: absolute;
    #box-label {
      width: 100%;
      color: white;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      font-size: 0.7rem;
    }
  }
  #cls {
    display: flex;
    flex-direction: column-reverse;
    position: absolute;
    width: 100%;
    height: 100%;
    #cls-label {
      color: white;
      text-align: center;
    }
  }
  img {
    flex-grow: 1;
    flex-shrink: 1;
  }
}
</style>
