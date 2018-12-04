<template>
  <component-frame :width-weight="6" :height-weight="4">
    <template slot="header-slot">
      Train Progress
    </template>
    <div class="progress wrap">
      <div class="bar" v-if="this.getFilteredModelList.length != 0">
        <section v-for="model in reduceModelList(this.getFilteredModelList)" :style="'width:' + calc_width(model[1]) + '%;background:'+ getAlgorithmColor(parseInt(model[0])) +';'">
          <div id="alg-title">
            {{ getAlgorithmTitleFromId(parseInt(model[0])) }}
          </div>
          <div id="alg-num">
            ({{ model[2] }})
          </div>
        </section>
      </div>
      <div class="bar" v-else>
        <section id="green" style="width: 100%">No model</section>
      </div>
      <div id="component-progress" class="scrollbar-container">
        <progress-bar :isTitle="true"/>
        <progress-bar v-for="item in getRunningModelList" :model="item"/>
      </div>
    </div>
  </component-frame>
</template>

<script>

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
        }, {})).map(d => [d[0], parseFloat(d[1]) / parseFloat(model_list.length), d[1]])
      return model_list
    },
    calc_width: function (width) {
      let percent = width * 100
      return percent
    }
  }
}
</script>

<style lang='scss' scoped>
  #component-progress {
    height: 70%;
    overflow: auto;
    line-height: auto;
  }
  .bar {
  display: flex;
	width: calc(100% - 40px);
	height: 30%;
	margin: 0 auto;
  padding-top:8%;
  margin-bottom:8%;
  margin-left: 20px;
  margin-right: 20px;
	overflow: hidden;
  }
  .bar section {
  	min-width: 10%;
    height: 30px;
    line-height: 30px;
    text-align: center;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    #alg-title {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 90%;
      height: 60%;
    }
    #alg-num {
      width: 100%;
      font-size: 70%;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 40%;
    }
  }
  .bar section:last-of-type {
    flex: auto;
  }
  #green {
    background: #65d260;
  } 
</style>
