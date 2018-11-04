<template>
  <component-frame :width-weight="4" :height-weight="4">
    <template slot="header-slot">
      Model Distribution
    </template>
    <div id="component-scatter">
      <div id="scatter-canvas">
      </div>
    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import * as d3 from 'd3'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentScatter',
  components: {
    'component-frame': ComponentFrame
  },
  created: function () {

  },
  mounted: function () {
    // Temtitively
    this.draw()
  },
  updated: function () {
    this.draw()
  },
  computed: {
    ...mapGetters(['getFilteredAndGroupedModelList'])
  },
  methods: {
    draw: function () {
      d3.select('svg').remove() // Remove SVG if it has been created.
      const margin = {top: 20, left: 45, right: 20, bottom: 30}
      const canvas = document.getElementById('scatter-canvas')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const model_list = this.getFilteredAndGroupedModelList
      const svg = d3.select('#scatter-canvas').append('svg')

      // Set size.
      svg
        .attr('width', canvas_width)
        .attr('height', canvas_height)

      // Axis Settings
      const scaleX = d3.scaleLinear().domain([0, 100])
        .range([0, canvas_width - margin.left - margin.right])
      const scaleY = d3.scaleLinear().domain([0, 100])
        .range([canvas_height - margin.bottom - margin.top, 0])
      const axX = d3.axisBottom(scaleX).ticks(5)
        .tickFormat((d, i) => {
          if (d === 100) {
            return d + ' [%]'
          } else {
            return d
          }
        })

      const axY = d3.axisLeft(scaleY).ticks(5)
        .tickFormat((d, i) => {
          if (d === 100) {
            return d + ' [%]'
          } else {
            return d
          }
        })
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, canvas_height - margin.bottom] + ')')
        .call(axX)
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .call(axY)

      // Sublines.
      // Horizontal
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line')
        .call(
          d3.axisRight()
            .tickSize(canvas_width - margin.left - margin.right)
            .tickFormat('')
            .scale(scaleY)
        )
      // Vertical
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line')
        .call(
          d3.axisTop()
            .tickSize(-canvas_height + margin.top + margin.bottom)
            .tickFormat('')
            .scale(scaleX)
        )

      // Plot Models.
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'model-circle')
        .selectAll('circle')
        .data(model_list)
        .enter()
        .append('circle')
        .attr('r', 6)
        .attr('fill', (d) => {
          return 'black'
        })
        .attr('cx', (d) => {
          // TODO: Modify data distribution
          const total_width = canvas_width - margin.left - margin.right
          const rescaled_point_x = d[0] * total_width
          return rescaled_point_x
        })
        .attr('cy', (d) => {
          // TODO: Modify data distribution
          const total_height = canvas_height - margin.top - margin.bottom
          const rescaled_point_y = (1 - d[1]) * total_height
          return rescaled_point_y
        })
        .on('mouseover', (d) => {
          // TODO: Fix event handler.
          d3.select(this).attr('fill')
        })
    }
  }
}
</script>

<style lang='scss'>
#component-scatter {
  height: 100%;
  padding: $scatter-padding;
  #scatter-canvas {
    width: 100%;
    height: 100%;
    .grid-line line {
      stroke: $scatter-grid-color;
    }
    .model-circle circle {
      r: $scatter-circle-radius;
    }
  }
}
</style>
