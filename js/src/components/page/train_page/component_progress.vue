<template>
  <component-frame :width-weight="6" :height-weight="4">
    <template slot="header-slot">
      Train Progress
    </template>
    <div id="alg-list">
      <div class="bar" v-if="this.getFilteredModelList">
        <section v-for="model in reduceModelList(this.getFilteredModelList)" :style="'width:' + calc_width(model[1]) + '%;background:'+ getAlgorithmColor(parseInt(model[0])) +';'">
          {{ getAlgorithmTitleFromId(parseInt(model[0])) }}
        </section>
      </div>
      <div class="bar" v-else>
        <section id="green" style="width: 100%">Green</section>
      </div>
    </div>
    <div id="component-progress" class="scrollbar-container">
      <progress-bar :isTitle="true"/>
      <progress-bar v-for="item in getRunningModelList" :model="item"/>
    </div>
  </component-frame>
</template>

<script>

import * as d3 from 'd3'
import { mapGetters } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ProgressBar from '@/components/page/train_page/progress_bar.vue'

export default {
  name: 'ComponentProgress',
  components: {
    'component-frame': ComponentFrame,
    'progress-bar': ProgressBar
  },
  computed: {
    ...mapGetters([
      'getRunningModelList',
      'getFilteredModelList',
      'getAlgorithmTitleFromId',
      'getAlgorithmColor'
    ]),
  },
  created: function () {
    window.addEventListener('resize', this.draw, false)
  },
  mounted: function () {
    // this.draw()
  },
  watch: {
    getFilteredModelList: function () {
      // this.draw()
    }
  },
  methods: {
    reduceModelList: function (model_list) {
      model_list = Object.entries(model_list.reduce(
        function (algs, model) {
          const id = model.algorithm_id
          if (id in algs) {
            algs[id] += 1
          } else {
            algs[id] = 1
          }
          return algs
        }, {})).map(d => [d[0], parseFloat(d[1]) / parseFloat(model_list.length)])
      return model_list
    },
    calc_width: function (width) {
      let percent = width * 100
      return percent
    }
  }
}
</script>

<style lang='scss'>
  #alg-list {
    width: 100%;
    height: 30%;
  }
  #component-progress {
    height: 70%;
    overflow: auto;
    line-height: auto;
  }
  .bar {
  display: flex;
	height: 30%;
	width: 100%;
	margin: 0 auto;
	
	overflow: hidden;
  }
  .bar section {
    height: 30px;
    line-height: 30px;
    text-align: center;
    color: white;
  }
  .bar section:last-of-type {
    flex: auto;
  }
  #green {
    background: #65d260;
  }
  #blue {
    background: #45afd9;
  }
  #yellow {
    background: #f6b50a;
  }
  #red {
    background: #f54954;
  }
</style>
