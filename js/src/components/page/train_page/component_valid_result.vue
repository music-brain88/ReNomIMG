<template>
  <component-frame :width-weight="8" :height-weight="7">
    <template slot="header-slot">
      Prediction Result
      <div id="valid-prediction-button-area"
        v-on:keyup.right="nextPage" v-on:keyup.left="prevPage" tabindex="0">
        <label>
          <input class="checkbox" type="checkbox"
          id="prediction-show-button" v-model="show_image" :disabled="!isTaskSegmentation">
          Image
        </label>
        <label>
          <input class="checkbox" type="checkbox" id="prediction-show-button" v-model="show_prediction">
          Prediction
        </label>
        <label>
          <input class="checkbox" type="checkbox" id="prediction-show-button" v-model="show_target">
          Target
        </label>
      </div>
    </template>
    <div id="pager" v-on:keyup.right="nextPage" v-on:keyup.left="prevPage" tabindex="0">
      <div class="pager-arrow" @click="prevPage">
        <i class="fa fa-caret-left" aria-hidden="true"></i>
      </div>
      <div class="pager-number" v-for="item in pageList()" @click="setPageNum(item)" :style="pagerStyle(item)">
        {{ item }}
      </div>
      <div class="pager-arrow" @click="nextPage">
        <i class="fa fa-caret-right" aria-hidden="true"></i>
      </div>
    </div>
    <div id="img-container" ref="container">
      <!--<transition-group name="fade">-->
      <div v-for="(item, index) in getValidImages" :style="getImgSize(item)" @click="showImageModal(item)">
        <img :src="item.img" v-if="showImage"/>
        <!--Change following div for each task-->
        <div id="cls" v-if='isTaskClassification'
          :style="getClassificationStyle(getValidResult(item))">
        </div>
        <div id="box" v-else-if='isTaskDetection'
          :style="getBoxStyle(box)" v-for="box in getValidResult(item)">
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
import { render_segmentation, setup_image_list } from '@/utils.js'
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentValidResult',
  components: {
    'component-frame': ComponentFrame
  },
  data: function () {
    return {
      show_image: true,
      show_target: false,
      show_prediction: true
    }
  },
  computed: {
    ...mapState([
      'datasets',
      'modal_image',
      'modal_prediction',
      'modal_target'
    ]),
    ...mapGetters(['getSelectedModel',
      'getCurrentTask',
      'getTagColor',
      'getImagePageOfPredictionSample'
    ]),
    showImage: function () {
      return this.show_image || !this.isTaskSegmentation
    },
    getValidImages: function () {
      const model = this.getSelectedModel
      if (!this.$refs.container) return []
      if (model) {
        let current_page = this.getImagePageOfPredictionSample
        const dataset = this.datasets.find(d => d.id === model.dataset_id)
        if (!dataset) return []

        if (dataset.page.length === 0) {
          // Setup image page if it has not been set.
          this.setUpValidImages()
        }

        // Clip page number.
        const max_page_num = dataset.page.length - 1
        const page_num = Math.max(Math.min(current_page, max_page_num), 0)
        this.setImagePageOfPredictionSample(page_num)
        current_page = this.getImagePageOfPredictionSample
        return dataset.page[current_page]
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
  beforeUpdate: function () {
    this.$nextTick(function () {
      if (this.isTaskSegmentation) {
        let index = 0
        if (!this.getValidImages) return
        for (let item of this.getValidImages) {
          this.getSegmentationStyle(this.getValidResult(item), index)
          index += 1
        }
      }
    })
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setImagePageOfPredictionSample', 'showModal', 'setImageModalData']),
    ...mapActions(['loadSegmentationTargetArray']),
    showImageModal: function (item) {
      let pred = null
      let targ = null
      if (this.isTaskClassification) {
        pred = this.getClassificationList(item)
      } else if (this.isTaskDetection) {
        pred = this.getBoxList(item)
      } else if (this.isTaskSegmentation) {
        pred = this.getClassificationList(item)
      }
      this.setImageModalData({
        'img': item.img,
        'prediction': pred,
        'target': targ,
      })
      this.showModal({'show_image': true})
    },
    pagerStyle: function (index) {
      const current_page = this.getImagePageOfPredictionSample
      if (current_page === index) {
        return {
          'background-color': '#063662',
          'color': 'white',
        }
      }
    },
    setPageNum: function (index) {
      if (index === '...') return

      const model = this.getSelectedModel
      if (!model) return

      const dataset = this.datasets.find(d => d.id === model.dataset_id)
      if (!dataset) return

      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfPredictionSample
      if (index === current_page) return
      this.setImagePageOfPredictionSample(Math.min(index, max_page_num))
    },
    nextPage: function () {
      const model = this.getSelectedModel
      if (!model) return

      const dataset = this.datasets.find(d => d.id === model.dataset_id)
      if (!dataset) return

      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfPredictionSample
      this.setImagePageOfPredictionSample(Math.min(current_page + 1, max_page_num))
    },
    prevPage: function () {
      const model = this.getSelectedModel
      if (!model) return

      const dataset = this.datasets.find(d => d.id === model.dataset_id)
      if (!dataset) return

      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfPredictionSample
      this.setImagePageOfPredictionSample(Math.max(current_page - 1, 0))
    },
    pageList: function () {
      const model = this.getSelectedModel
      if (!model) return []

      const dataset = this.datasets.find(d => d.id === model.dataset_id)
      if (!dataset) return []
      if (!dataset.page) return []

      const current_page = Math.max(this.getImagePageOfPredictionSample, 0)
      const max_page_num = Math.max(dataset.page.length - 1, 0)

      if (max_page_num > 5) {
        if (current_page < 4) {
          return [...[...Array(Math.max(current_page, 5)).keys()], '...', max_page_num]
        } else if (current_page > max_page_num - 4) {
          return [0, '...', ...[...Array(Math.max(max_page_num - current_page, 5)).keys()].reverse().map(i => max_page_num - i)]
        } else {
          return [0, '...', ...[...Array(5).keys()].reverse().map(i => current_page - i + 2), '...', max_page_num]
        }
      } else {
        return Array(max_page_num).keys()
      }
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
      const parent_div = document.getElementById('img-container')
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
    setUpValidImages: function () {
      const parent_div = document.getElementById('img-container')
      if (!parent_div) return
      const parent_height = parent_div.clientHeight
      const parent_width = parent_div.clientWidth
      const child_margin = Math.min(this.vh(0.25), this.vw(0.25))

      const model = this.getSelectedModel
      if (!model) return

      const dataset = this.datasets.find(d => d.id === model.dataset_id)
      if (!dataset) return

      // Using vue-worker here.
      // See https://github.com/israelss/vue-worker
      this.$worker.run(setup_image_list, [dataset.valid_data, parent_width, parent_height, child_margin])
        .then((ret) => {
          dataset.page = ret
        })
    },
    getValidResult: function (item) {
      if (item.index < 0) return
      const index = item.index
      const model = this.getSelectedModel
      if (!model) return []
      let result = []

      if (this.show_target) {
        const dataset = this.datasets.find(d => d.id === model.dataset_id)
        result = dataset.getValidTarget(index)
      }
      if (this.show_prediction) {
        if (this.isTaskClassification) {
          result = model.getValidResult(index)
        } else if (this.isTaskDetection) {
          result = result.concat(model.getValidResult(index))
        } else if (this.isTaskSegmentation) {
          result = model.getValidResult(index)
        }
      }
      return result
    },
    getClassificationStyle: function (cls) {
      if (!cls) {
        return {}
      }
      if (cls.hasOwnProperty('score') && cls.hasOwnProperty('class')) {
        const class_id = cls.class
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
    getSegmentationStyle: function (item, index) {
      if (!item || !this.show_prediction) {
        // Clear canvas
        var canvas = document.getElementById('canvas-' + String(index))
        var cxt = canvas.getContext('bitmaprenderer')
        var offCanvas = new OffscreenCanvas(canvas.width, canvas.height)
        var offCxt = offCanvas.getContext('2d')
        offCxt.clearRect(0, 0, canvas.width, canvas.height)
        cxt.transferFromImageBitmap(offCanvas.transferToImageBitmap())
        return
      }
      this.$worker.run(render_segmentation, [item]).then((ret) => {
        var canvas = document.getElementById('canvas-' + String(index))
        var cxt = canvas.getContext('bitmaprenderer')
        cxt.transferFromImageBitmap(ret)
      })
    },
    getSegmentationTargetArray: function (src) {
      let arr = this.loadSegmentationTargetArray(src)
    }
  }
}
</script>

<style lang='scss'>
#valid-prediction-button-area {
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

#img-container{
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
  justify-content: flex-end;
  padding-right: 5px;
  align-items: center;
  .pager-arrow {
    font-size: 150%;
    height: 100%;
    display: flex;
    align-items: center;
    margin-left: 5px;
    margin-right: 5px;
    cursor: pointer;
    i {
    }
  }
  .pager-number {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 75%;
    height: calc(100% - 2px);
    width: 3%;
    cursor: pointer;
    letter-spacing: -1px;
    transition: all 0.1s
  }
}
</style>
