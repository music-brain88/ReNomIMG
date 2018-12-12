<template>
  <component-frame :width-weight="8" :height-weight="7">

    <!--Header Contents--------------->
    <template slot="header-slot">
      Prediction Result
      <div id="valid-prediction-button-area" tabindex="0"
        v-on:keyup.right="nextPage" v-on:keyup.left="prevPage">
        <!--
          Only if Segmentation, show image toggle will be shown.
        -->
        <label v-if="isTaskSegmentation">
          <input class="checkbox" type="checkbox"
          id="prediction-show-button" v-model="show_image" :disabled="!isTaskSegmentation">
          Image
        </label>

        <!--
          'Show prediction result' Button and 'show target' Button.
          - In classification, prediction or target can be on.
          - In detection, both prediction and target can be on.
          - In segmentation, prediction or target can be on.
        -->
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
    <!---------------Header Contents-->


    <!--Pager Settings--------------->
    <div id="pager" tabindex="0"
      v-on:keyup.right="nextPage" v-on:keyup.left="prevPage">

      <!--Left Arrow-->
      <div class="pager-arrow" @click="prevPage">
        <i class="fa fa-caret-left" aria-hidden="true"></i>
      </div>

      <!--Number-->
      <div v-for="item in pageList()" class="pager-number" 
        :class="{number: item !== '...'}"  @click="setPageNum(item)" :style="pagerStyle(item)">
        {{ item }}
      </div>

      <!--Right Arrow-->
      <div class="pager-arrow" @click="nextPage">
        <i class="fa fa-caret-right" aria-hidden="true"></i>
      </div>
    </div>
    <!---------------Pager Settings-->


    <!--Image list--------------->
    <div id="img-container" ref="container">
      <div v-for="(item, index) in getValidImages" :style="getImgSize(item)" @click="showImageModal(item)">
        <img :src="item.img" v-if="showImage"/>

        <!--Classification-->
        <div id="cls" v-if='isTaskClassification'
          :style="getClassificationStyle(getValidResult(item))">
        </div>

        <!--Detection-->
        <div id="box" v-else-if='isTaskDetection'
          :style="getBoxStyle(box)" v-for="box in getValidResult(item)">
          <div id="box-label" v-if="box" :style="getBoxLabelColor(box.class)">&nbsp&nbsp{{box.name}}</div>
        </div>

        <!--Segmentation-->
        <div id="seg" v-else-if='isTaskSegmentation'>
          <canvas ref="canvas"/>
        </div>
      </div>
    </div>
    <!---------------Image list-->
  </component-frame>
</template>

<script>
import { TASK_ID } from '@/const.js'
import { getTagColor, render_segmentation, setup_image_list } from '@/utils.js'
import { mapGetters, mapState, mapMutations, mapActions } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentValidResult',
  components: {
    'component-frame': ComponentFrame
  },
  data: function () {
    return {
      // The state of the checkbox which represents weather image is shown.
      show_image: true,
      // The state of the checkbox which represents weather target is shown.
      show_target: false,
      // The state of the checkbox which represents weather prediction is shown.
      show_prediction: true
    }
  },
  beforeUpdate: function () {
    /**
      If the task is segmentation, drawing function will be called in
      each update.
    */
    this.$nextTick(function () {
      if (this.isTaskSegmentation) {
        if (!this.getValidImages) return
        let canvas_index = 0
        for (let item of this.getValidImages) {
          // This function will draw item to the canvas witch has same index to canvas_index.
          this.getValidResult(item, canvas_index)
          canvas_index += 1
        }
      }
    })
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
      if (!this.$refs.container === undefined) return

      const dataset = this.dataset
      if (!dataset) return
      if (dataset.page.length === 0) {
        // Setup image page if it has not been set.
        this.setUpValidImages()
      }

      // Clip page number.
      let current_page = this.getImagePageOfValid
      const max_page_num = dataset.page.length - 1
      const page_num = Math.max(Math.min(current_page, max_page_num), 0)
      this.setImagePageOfValid(page_num)
      current_page = this.getImagePageOfValid
      return dataset.page[current_page]
    },
  },
  methods: {
    ...mapMutations([
      'setImagePageOfValid', // Set current page number.
      'showModal',
      'setImageModalData' // This will set index of image for show in modal.
    ]),
    ...mapActions([
      'loadSegmentationTargetArray' // Get segmentation target from server.
    ]),
    showImageModal: function (item) {
      /**
        The image modal will appear.
      */
      this.setImageModalData(item.index)
      this.showModal({'show_image': true})
    },
    pagerStyle: function (index) {
      /**
        This returns pager number style.
        The current page number will be emphasized.
      */
      const current_page = this.getImagePageOfValid
      if (current_page === index) {
        return {
          'background-color': '#063662',
          'color': 'white',
        }
      }
    },
    setPageNum: function (index) {
      /**
        If the pushed pager button is number,
        set current page number as new number.
      */
      if (index === '...') return
      const dataset = this.dataset
      if (!dataset) return

      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfValid
      if (index === current_page) return
      this.setImagePageOfValid(Math.min(index, max_page_num))
    },
    nextPage: function () {
      /**
        Go to next page.
      */
      const dataset = this.dataset
      if (!dataset) return
      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfValid
      this.setImagePageOfValid(Math.min(current_page + 1, max_page_num))
    },
    prevPage: function () {
      /**
        Go to previous page.
      */
      const dataset = this.dataset
      if (!dataset) return
      const max_page_num = dataset.page.length - 1
      const current_page = this.getImagePageOfValid
      this.setImagePageOfValid(Math.max(current_page - 1, 0))
    },
    pageList: function () {
      /**
        Get the pager list.
      */
      const dataset = this.dataset
      if (!dataset) return
      if (!dataset.page) return

      const current_page = Math.max(this.getImagePageOfValid, 0)
      const max_page_num = Math.max(dataset.page.length - 1, 0)

      if (max_page_num > 5) {
        if (current_page < 5) {
          return [...[...Array(Math.max(current_page, 7)).keys()], '...', max_page_num]
        } else if (current_page > max_page_num - 5) {
          return [0, '...', ...[...Array(Math.max(max_page_num - current_page, 7)).keys()].reverse().map(i => max_page_num - i)]
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
      /**
        Get the size of image.
      */
      const parent_div = this.$refs.container
      if (parent_div === undefined) return
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

      // Using vue-worker here.
      // See https://github.com/israelss/vue-worker
      this.$worker.run(setup_image_list, [dataset.valid_data, parent_width, parent_height, child_margin])
        .then((ret) => {
          dataset.page = ret
        })
    },
    getValidResult: function (item, canvas_index = 0) {
      /**
        This function gets the result of prediction or target.
        If the task is segmentation, this will call draw function.

        Args:
          item: This is a item of dataset.valid_data. It has following format.
            {
              index: (nth image),
              img: (url),
              size: (image size)
            }
          canvas_index: Required for drawing segmentation result.
            In other task, this is not used.
      */

      if (item.index < 0) return
      const index = item.index
      const model = this.getSelectedModel
      if (!model) return
      let result
      if (this.show_target) {
        const dataset = this.dataset
        if (!dataset) return
        result = dataset.getValidTarget(index)
        if (this.isTaskClassification) {
          return result
        } else if (this.isTaskSegmentation) {
          return this.loadSegmentationTargetArray({
            name: result.name,
            size: [
              parseInt(model.hyper_parameters.imsize_w),
              parseInt(model.hyper_parameters.imsize_h)],
            callback: (response) => {
              const item = response.data
              this.getSegmentationStyle(item, canvas_index)
            }
          })
        }
      }
      if (this.show_prediction) {
        if (this.isTaskClassification) {
          return model.getValidResult(index)
        } else if (this.isTaskDetection) {
          if (!result) result = []
          result = result.concat(model.getValidResult(index))
        } else if (this.isTaskSegmentation) {
          result = model.getValidResult(index)
          this.getSegmentationStyle(result, canvas_index)
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
          border: 'solid 1.5px' + getTagColor(class_id) + 'bb'
        }
      } else {
        const class_id = cls
        return {
          border: 'solid 1.5px' + getTagColor(class_id) + 'bb'
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
        border: 'solid 2px' + getTagColor(class_id) + 'bb'
      }
    },
    getBoxLabelColor: function (class_id) {
      return {
        'background-color': getTagColor(class_id) + 'bb'
      }
    },
    getSegmentationStyle: function (item, index) {
      if (!item) return
      if (!item || !this.show_prediction) {
        // Clear canvas
        var canvas = this.$refs.canvas[index]
        if (!canvas) return
        var cxt = canvas.getContext('bitmaprenderer')
        var offCanvas = new OffscreenCanvas(canvas.width, canvas.height)
        var offCxt = offCanvas.getContext('2d')
        offCxt.clearRect(0, 0, canvas.width, canvas.height)
        cxt.transferFromImageBitmap(offCanvas.transferToImageBitmap())
        return
      }
      this.$worker.run(render_segmentation, [item]).then((ret) => {
        var canvas = this.$refs.canvas[index]
        var cxt = canvas.getContext('bitmaprenderer')
        cxt.transferFromImageBitmap(ret)
      })
    },
  }
}
</script>

<style lang='scss'>
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
      #box-label {
        display: flex;
        min-width: 100%;
        height: calc(20px - 2.5px);
        position: relative;
        background-color: white;
        color: white;
        font-size: 0.8rem;
        margin: 0;
      }
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
    color: lightgray;
    transition: all 0.02s;
    i {
    }
    &:hover {
      color: gray;
    }
  }
  .pager-number {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 75%;
    height: calc(100% - 3px);
    margin-top: 2px;
    width: 3%;
    letter-spacing: -1px;
    color: gray;
  }
  .number {
    transition: all 0.1s;
    cursor: pointer;
    &:hover {
      color: black;
    }
  }
}
</style>
