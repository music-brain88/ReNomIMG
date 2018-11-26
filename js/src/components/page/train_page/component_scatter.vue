<template>
  <component-frame :width-weight="4" :height-weight="4">
    <template slot="header-slot">
      Model Distribution
    </template>
    <div id="component-scatter">
      <div id="title-metric1">
        {{ getTitleMetric1 }}
      </div>
      <div id="title-metric2">
        {{ getTitleMetric2 }}
      </div>
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
  data: function () {
    return {
      componentWidth: window.innerWidth,
      componentHeight: window.innerHeight,
    }
  },
  created: function () {
    // Register function to window size listener
    // TODO: Need to use lodash here.
    window.addEventListener('resize', this.draw, false)
  },
  mounted: function () {
    this.draw()
  },
  beforeDestroy: function () {
    // Remove it.
    window.removeEventListener('resize', this.draw, false)
  },
  updated: function () {
    this.draw()
  },
  computed: {
    ...mapGetters([
      'getFilteredAndGroupedModelList',
      'getColorClass',
      'getAlgorithmColor',
      'getTitleMetric1',
      'getTitleMetric2'
    ])
  },
  watch: {
    getFilteredAndGroupedModelList: function () {
      this.draw()
    }
  },
  methods: {
    draw: function () {
      if (!this.getFilteredAndGroupedModelList) return
      d3.select('#scatter-canvas').select('svg').remove() // Remove SVG if it has been created.

      const margin = {top: 15, left: 40, right: 20, bottom: 20}
      const canvas = document.getElementById('scatter-canvas')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const circle_radius = Math.min(canvas_width * 0.02, canvas_height * 0.02)
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
      const axY = d3.axisLeft(scaleY).ticks(5)

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

      let tooltip = d3.select('#scatter-canvas')
        .append('div')
        .style('display', 'none')

      // Plot Models.
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .selectAll('circle')
        .data(model_list)
        .enter()
        .append('circle')
        .attr('r', circle_radius)
        .attr('class', (m) => { return this.getColorClass(m) })
        .attr('cx', (m) => {
          // TODO: Modify data distribution
          const metric = m.getResultOfMetric1()
          if (metric.value === '-') metric.value = 0
          const total_width = canvas_width - margin.left - margin.right
          const rescaled_point_x = metric.value * total_width
          return rescaled_point_x
        })
        .attr('cy', (m) => {
          // TODO: Modify data distribution
          const metric = m.getResultOfMetric2()
          if (metric.value === '-') metric.value = 0
          const total_height = canvas_height - margin.top - margin.bottom
          const rescaled_point_y = (1 - metric.value) * total_height
          return rescaled_point_y
        })
        .attr('fill', (m) => {
          return this.getAlgorithmColor(m.algorithm_id)
        })
        .on('mouseenter', (d, index) => {
          // TODO: Fix event handler.
          tooltip.style('display', 'inline-block')
          tooltip.transition()
            .duration(200)
            .style('opacity', 0.9)
          tooltip.html(
            'model_id:' + d.id + '<br />' +
            'mAP:' + d.best_epoch_valid_result.mAP + '<br />' +
            'IoU:' + d.best_epoch_valid_result.IOU + '<br />'
          )
            .style('position', 'absolute')
            // .style('width', '100px')
            // .style('height', '50px')
            .style('top', (d3.event.pageY - 28) + 'px')
            .style('left', (d3.event.pageX) + 'px')
            .style('padding', '10px')
            .style('background', this.getAlgorithmColor(d.algorithm_id))
            .style('color', 'white')
            .style('text-align', 'left')
            .on('mouseenter', function () {
              tooltip.style('display', 'inline-block')
            })
            .on('mouseleave', function () {
              tooltip.style('display', 'none')
            })
        })
        .on('mouseleave', function () {
          tooltip.style('display', 'none')
          tooltip.style('opacity', 0)
        })
    }
  }
}
</script>

<style lang='scss'>
#component-scatter {
  height: 100%;
  width: 100%;
  padding: $scatter-padding;
  position: relative;
  #title-metric1 {
    position: absolute;
    top: calc(100% - #{$scatter-padding});
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding});
    height: $scatter-padding;
    text-align: center;
    font-size: 70%;
  }
  #title-metric2 {
    position: absolute;
    top: 0;
    left: calc(#{$scatter-padding}*0.7);
    width: $scatter-padding;
    height: 100%;
    writing-mode: vertical-rl;
    text-align: center;
    font-size: 70%;
  }
  #scatter-canvas {
    position: absolute;
    top: $scatter-padding;
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding}*2);
    height: calc(100% - #{$scatter-padding}*2);
    .grid-line line {
      stroke: $scatter-grid-color;
    }
  }
  div.tooltip {
  position: absolute;
  text-align: center;
  width: 60px;
  height: 28px;
  padding: 2px;
  font: 12px sans-serif;
  background: lightsteelblue;
  border: 0px;
  border-radius: 8px;
  pointer-events: none;
  }
}
</style>
