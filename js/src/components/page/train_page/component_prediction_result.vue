<template>
  <component-frame :width-weight="8" :height-weight="7">
    <template slot="header-slot">
      Prediction Result
    </template>
    <div id="pager">
      <div class="pager-arrow" @click="()=>{count ++}">
        <i class="fa fa-caret-left" aria-hidden="true"></i>
      </div>
      <div class="pager-number" v-for="item in [1, 2, 3]">
        {{item}}
      </div>
      <div class="pager-arrow">
        <i class="fa fa-caret-right" aria-hidden="true"></i>
      </div>
    </div>
    <div id="img-container">
      <!--<transition-group name="fade">-->
      <div v-for="item in getValidImages" :style="getImgSize(item)">
        <img :src="item.img"/>
        <div id="box" v-if='isTaskDetection' :style="getBoxStyle(box)" v-for="box in getBoxList(item)">
        </div>
      </div>
      <!--</transition-group>-->
    </div>
  </component-frame>
</template>

<script>
import { TASK_ID } from '@/const.js'
import { mapGetters, mapState } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentPredictionResult',
  components: {
    'component-frame': ComponentFrame
  },
  data: function () {
    return {
      count: 0,
      show_target: false
    }
  },
  computed: {
    ...mapState(['datasets']),
    ...mapGetters(['getSelectedModel', 'getCurrentTask', 'getTagColor']),
    getValidImages: function () {
      const model = this.getSelectedModel
      if (model) {
        const dataset = this.datasets.find(d => d.id === model.dataset_id)
        if (dataset && dataset.page.length === 0) {
          // Setup image page if it has not been set.
          this.setUpValidImages()
        }
        return dataset.page[this.count]
      }
      return []
    },
    isTaskDetection: function () {
      return this.getCurrentTask === TASK_ID.DETECTION
    }
  },
  created: function () {

  },
  methods: {
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

      function setup (dataset, parent_width, parent_height, margin) {
        const brank = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
        const valid_data = dataset.valid_data
        const pages = []
        const last_index = valid_data.img.length - 1
        let one_page = []
        let nth_page = 0
        let nth_line_in_page = 0
        let accumurated_ratio = 0
        let max_ratio = (parent_width / (parent_height / 3))
        for (let i = 0; i < valid_data.size.length; i++) {
          let size = valid_data.size[i]
          let ratio = ((size[0] + 2 * margin) / (size[1] + 2 * margin))
          accumurated_ratio += ratio
          if (accumurated_ratio <= max_ratio) {
            one_page.push({index: i, img: valid_data.img[i], size: valid_data.size[i]})
          } else {
            if (nth_line_in_page >= 2) {
              pages.push(one_page)
              nth_page++
              one_page = [{index: i, img: valid_data.img[i], size: valid_data.size[i]}]
              accumurated_ratio = ratio
              nth_line_in_page = 0
            } else {
              one_page.push({index: i, img: valid_data.img[i], size: valid_data.size[i]})
              accumurated_ratio = ratio
              nth_line_in_page++
            }
          }
          if (i === last_index) {
            // Add white image to empty space.
            console.log(last_index, i, nth_line_in_page)
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
          dataset.page = ret
        })
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
      const model = this.getSelectedModel
      if (!model) return
      let box_list = []
      if (this.show_target) {
        box_list = model.best_epoch_valid_result.target_box[item.index]
      } else {
        box_list = model.best_epoch_valid_result.prediction_box[item.index]
      }
      return box_list
    }
  }
}
</script>

<style lang='scss'>
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
