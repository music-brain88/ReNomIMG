<template>
  <component-frame :width-weight="9" :height-weight="9">
    <template slot="header-slot">
      Prediction Result
    </template>
    <div id="pager" v-on:keyup.right="nextPage" v-on:keyup.left="prevPage" tabindex="0">
      <div class="pager-arrow" @click="prevPage">
        <i class="fa fa-caret-left" aria-hidden="true"></i>
      </div>
      <div class="pager-number" v-for="item in pageList()">
        {{item}}
      </div>
      <div class="pager-arrow" @click="nextPage">
        <i class="fa fa-caret-right" aria-hidden="true"></i>
      </div>
    </div>
    <div id="prediction-img-container" ref="container">
      <!--<transition-group name="fade">-->
      <div v-for="(item, index) in getValidImages" :style="getImgSize(item)">
        <img :src="item.img"/>
        <!--Change following div for each task-->
        <div id="cls" v-if='isTaskClassification'
          :style="getClassificationStyle(getClassificationList(item))">
        </div>
        <div id="box" v-else-if='isTaskDetection'
          :style="getBoxStyle(box)" v-for="box in getBoxList(item)">
        </div>
        <div id="seg" v-else-if='isTaskSegmentation'>
          <canvas :id="'canvas-' + index"/>
        </div>
      </div>
      <!--</transition-group>-->
    </div>
  </component-frame>
</template>

<script>
import { TASK_ID } from '@/const.js'
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentPrediction',
  components: {
    'component-frame': ComponentFrame
  },
  data: function () {
    return {
      show_target: false,
      prediction_result: []
    }
  },
  computed: {
    ...mapState(['datasets']),
    ...mapGetters(['getDeployedModel',
      'getCurrentTask',
      'getTagColor',
      'getImagePageOfPrediction'
    ]),
    getValidImages: function () {
      const model = this.getDeployedModel
      if (model) {
        let current_page = this.getImagePageOfPrediction
        const dataset = this.prediction_result
        if (dataset.length === 0) {
          // Setup image page if it has not been set.
          this.setUpImages()
        }

        // Clip page number.
        const max_page_num = dataset.length - 1
        const page_num = Math.max(Math.min(current_page, max_page_num), 0)
        this.setImagePageOfPrediction(page_num)
        current_page = this.getImagePageOfPrediction
        return dataset[current_page]
      }
      return []
    },
    isTaskClassification: function () {
      return this.getCurrentTask === TASK_ID.CLASSIFICATION
    },
    isTaskDetection: function () {
      return this.getCurrentTask === TASK_ID.DETECTION
    },
    isTaskSegmentation: function () {
      return this.getCurrentTask === TASK_ID.SEGMENTATION
    }
  },
  updated: function () {
    this.$nextTick(function () {
      if (this.isTaskSegmentation) {
        let index = 0
        if (!this.getValidImages) return
        for (let item of this.getValidImages) {
          this.getSegmentationStyle(this.getSegmentationList(item), index)
          index += 1
        }
      }
    })
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setImagePageOfPrediction']),
    ...mapActions(['loadSegmentationTargetArray']),
    nextPage: function () {
      const model = this.getDeployedModel
      if (!model) return

      const dataset = this.prediction_result
      if (!dataset) return

      const max_page_num = dataset.length - 1
      const current_page = this.getImagePageOfPrediction
      this.setImagePageOfPrediction(Math.min(current_page + 1, max_page_num))
    },
    prevPage: function () {
      const model = this.getDeployedModel
      if (!model) return

      const dataset = this.prediction_result
      if (!dataset) return

      const max_page_num = dataset.length - 1
      const current_page = this.getImagePageOfPrediction
      this.setImagePageOfPrediction(Math.max(current_page - 1, 0))
    },
    pageList: function () {
      const model = this.getDeployedModel
      if (!model) return []

      const dataset = this.prediction_result
      if (!dataset) return []

      const current_page = this.getImagePageOfPrediction
      const max_page_num = dataset.length - 1
      // TODO: Return page numbers.
      return [0, 1, 2, '...', max_page_num]
    },
    vh: function (v) {
      var h = Math.max(document.documentElement.clientHeight, window.innerHeight || 0)
      return (v * h) / 100
    },
    vw: function (v) {
      var w = Math.max(document.documentElement.clientWidth, window.innerWidth || 0)
      return (v * w) / 100
    },
    getImgSize: function (item) {
      const parent_div = document.getElementById('prediction-img-container')
      if (!parent_div) return {}
      const parent_height = parent_div.clientHeight
      const child_margin = Math.min(this.vh(0.25), this.vw(0.25))
      const height = (parent_height - child_margin * 6) / 3
      const width = item.size[0] / item.size[1] * height
      return {
        height: height + 'px',
        width: width + 'px',
      }
    },
    setUpImages: function () {
      const parent_div = document.getElementById('prediction-img-container')
      if (!parent_div) return
      const parent_height = parent_div.clientHeight
      const parent_width = parent_div.clientWidth
      const child_margin = Math.min(this.vh(0.25), this.vw(0.25))

      const model = this.getDeployedModel
      if (!model) return

      const dataset = model.prediction_result
      if (!dataset) return

      function setup (dataset, parent_width, parent_height, margin) {
        const brank = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
        const predicted_data = dataset
        const pages = []
        const last_index = predicted_data.img.length - 1
        let one_page = []
        let nth_page = 0
        let nth_line_in_page = 1
        let accumurated_ratio = 0
        let max_ratio = (parent_width / (parent_height / 3))
        for (let i = 0; i < predicted_data.size.length; i++) {
          let size = predicted_data.size[i]
          let ratio = ((size[0] + 2 * margin) / (size[1] + 2 * margin))
          accumurated_ratio += ratio
          if (accumurated_ratio <= max_ratio || one_page.length === 0) {
            one_page.push({index: i, img: predicted_data.img[i], size: predicted_data.size[i]})
          } else {
            if (nth_line_in_page >= 3) {
              pages.push(one_page)
              nth_page++
              one_page = [{index: i, img: predicted_data.img[i], size: predicted_data.size[i]}]
              accumurated_ratio = ratio
              nth_line_in_page = 1
            } else {
              one_page.push({index: i, img: predicted_data.img[i], size: predicted_data.size[i]})
              accumurated_ratio = ratio
              nth_line_in_page++
            }
          }
          if (i === last_index) {
            // Add white image to empty space.
            one_page.push({index: -1, img: brank, size: [max_ratio - accumurated_ratio, 1]})
            for (let j = nth_line_in_page; j < 2; j++) {
              one_page.push({index: -1, img: brank, size: [max_ratio, 1]})
            }
          }
        }
        if (pages[pages.length - 1] !== one_page) {
          pages.push(one_page)
        }
        return pages
      }

      // Using vue-worker here.
      // See https://github.com/israelss/vue-worker
      this.$worker.run(setup, [dataset, parent_width, parent_height, child_margin])
        .then((ret) => {
          this.prediction_result = ret
        })
    },
    getClassificationList: function (item) {
      if (item.index < 0) return
      const model = this.getDeployedModel
      if (!model) return []
      if (!model.prediction_result) return null
      if (!model.prediction_result.prediction) return null
      return model.prediction_result.prediction[item.index]
    },

    getClassificationStyle: function (cls) {
      if (cls === null) {
        return {}
      }
      if (cls.hasOwnProperty('class') && cls.hasOwnProperty('score')) {
        const class_id = cls['class']
        return {
          border: 'solid 2.5px' + this.getTagColor(class_id) + 'bb'
        }
      } else {
        const class_id = cls
        return {
          border: 'solid 2.5px' + this.getTagColor(class_id) + 'bb'
        }
      }
    },
    getBoxStyle: function (box) {
      const class_id = box.class
      const x1 = (box.box[0] - box.box[2] / 2) * 100
      const y1 = (box.box[1] - box.box[3] / 2) * 100
      return {
        top: y1 + '%',
        left: x1 + '%',
        width: box.box[2] * 100 + '%',
        height: box.box[3] * 100 + '%',
        border: 'solid 2.5px' + this.getTagColor(class_id) + 'bb'
      }
    },
    getBoxList: function (item) {
      if (item.index < 0) return
      const model = this.getDeployedModel
      if (!model) return []

      let box_list = []
      if (!model.prediction_result) return []
      box_list = model.prediction_result.prediction[item.index]
      return box_list
    },
    getSegmentationList: function (item) {
      if (item.index < 0) return
      const model = this.getDeployedModel
      if (!model) return []
      if (!model.prediction_result) return null
      const pred = model.prediction_result.prediction[item.index]
      return pred
    },

    getSegmentationStyle: function (item, index) {
      if (!item) {
        var canvas = document.getElementById('canvas-' + String(index))
        var cxt = canvas.getContext('2d')
        cxt.clearRect(0, 0, canvas.width, canvas.height)
        return
      }
      let indexRow = 0
      let indexCol = 0
      const height = item.length
      const width = item[0].length

      var canvas = document.getElementById('canvas-' + String(index))
      if (!canvas) return
      var cxt = canvas.getContext('2d')
      canvas.height = height
      canvas.width = width
      cxt.clearRect(0, 0, width, height)
      for (let row of item) {
        indexCol = 0
        for (let col of row) {
          const color = this.getTagColor(col)
          if (col === 0) {
            cxt.fillStyle = color + '00'
          } else {
            cxt.fillStyle = color + '88'
          }
          cxt.fillRect(indexCol, indexRow, 1, 1)
          indexCol++
        }
        indexRow++
      }
    },
    getSegmentationTargetArray: function (src) {
      let arr = this.loadSegmentationTargetArray(src)
    }
  }
}
</script>

<style lang='scss'>
#prediction-img-container{
  width: 100%;
  height: 95%;
  display: flex;
  flex-wrap: wrap;
  div {
    display: inline-block;
    flex-grow: 1;
    flex-shrink: 1;
    overflow: hidden;
    position: relative;
    margin: 0.25vmin;
    img {
      width: 100%;
      height: 100%;
    }
    #box {
      position: absolute;
      height: 100%;
      width: 100%;
    }
    #cls {
      position: absolute;
      height: 100%;
      width: 100%;
      top: -0.25vmin;
      left: -0.25vmin;
    }
    #seg {
      position: absolute;
      height: 100%;
      width: 100%;
      top: -0.25vmin;
      left: -0.25vmin;
      canvas {
        height: 100%;
        width: 100%;
      }
    }
  }
}
#pager {
  width: 100%;
  height: 5%;
  display: flex;
  align-items: right;
  .pager-arrow {
    height: 100%;
    display: flex;
    align-items: center;
    margin: 1px;
    cursor: pointer;
    i {
    }
  }
  .pager-number {
    display: flex;
    align-items: center;
    height: 100%;
    border: solid 1px gray;
    margin: 1px;
    cursor: pointer;
  }
}
</style>
