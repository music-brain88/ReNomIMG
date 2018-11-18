<template>
  <component-frame :width-weight="6" :height-weight="4">
    <template slot="header-slot">
      Train Progress
    </template>
    <div id="alg-list">
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
    ...mapGetters(['getRunningModelList', 'getFilteredModelList', 'getAlgorithmTitleFromId']),
  },
  created: function () {
    window.addEventListener('resize', this.draw, false)
  },
  mounted: function () {
    this.draw()
  },
  watch: {
    getFilteredModelList: function () {
      this.draw()
    }
  },
  methods: {
    draw: function () {
      d3.select('svg').remove() // Remove SVG if it has been created.
      const margin = { top: 15, left: 45, right: 20, bottom: 20 }
      const canvas = document.getElementById('alg-list')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const circle_radius = Math.min(canvas_width * 0.03, canvas_height * 0.03)
      const svg = d3.select('#alg-list').append('svg')
      let model_list = this.getFilteredModelList
      const num_model = model_list.length

      model_list = Object.entries(model_list.reduce(
        function (algs, model) {
          const id = model.algorithm_id
          if (id in algs) {
            algs[id] += 1
          } else {
            algs[id] = 1
          }
          return algs
        }, {})).map(d => [d[0], parseFloat(d[1]) / parseFloat(num_model)])

      // Set size.
      svg
        .attr('width', canvas_width)
        .attr('height', canvas_height)

      // Plot Rectangle
      svg.append('g')
        .selectAll('circle')
        .data(model_list)
        .enter()
        .append('rect')
        .attr('x', (d, i) => {
          return d[1]
        })
        .attr('y', canvas_height * 0.5)
        .attr('width', (d) => d[1] * canvas_width)
        .attr('height', 25)
        .attr('color', 'blue')
    }
  }
}
</script>

<style lang='scss'>
  #alg-list {
    width: 100%;
    height: 30%;
    margin-bottom: 5px;
  }
  #component-progress {
    height: 70%;
    overflow: auto;
    line-height: auto;
  }
</style>
