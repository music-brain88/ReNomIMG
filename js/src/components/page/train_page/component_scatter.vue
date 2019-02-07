<template>
  <component-frame
    :width-weight="4"
    :height-weight="4">
    <template slot="header-slot">
      Model Distribution
    </template>
    <div id="component-scatter">
      <div id="title-metric1">
        {{ getTitleMetric1 }} [%]
      </div>
      <div id="title-metric2">
        {{ getTitleMetric2 }} [%]
      </div>
      <div id="scatter-canvas"/>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapMutations } from 'vuex'
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
      tooltip: null
    }
  },
  computed: {
    ...mapGetters([
      'getFilteredAndGroupedModelList',
      'getColorClass',
      'getAlgorithmColor',
      'getAlgorithmTitleFromId',
      'getTitleMetric1',
      'getTitleMetric2',
      'getSelectedModel'
    ])
  },

  watch: {
    getFilteredAndGroupedModelList: function () {
      this.draw()
    },
    getSelectedModel: function () {
      const svg = d3.select('#scatter-canvas').select('svg')
      if (!svg) return
      const canvas = document.getElementById('scatter-canvas')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const circle_radius = Math.min(canvas_width * 0.02, canvas_height * 0.02)
      svg.selectAll('circle')
        .attr('r', (m) => {
          const model = this.getSelectedModel
          if (model === m) {
            return circle_radius * 1.6
          } else {
            return circle_radius
          }
        })
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
  methods: {
    ...mapMutations(['setSelectedModel']),
    draw: function () {
      if (!this.getFilteredAndGroupedModelList) return
      d3.select('#scatter-canvas').select('svg').remove() // Remove SVG if it has been created.

      const margin = { top: 15, left: 40, right: 20, bottom: 35 }
      const canvas = document.getElementById('scatter-canvas')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const circle_radius = Math.min(canvas_width * 0.02, canvas_height * 0.02)
      const model_list = this.getFilteredAndGroupedModelList
      const svg = d3.select('#scatter-canvas').append('svg').attr('id', 'model-scatter-graph')

      // Set size.
      svg
        .attr('width', canvas_width)
        .attr('height', canvas_height)

      // Axis Settings
      const scaleX = d3.scaleLinear().domain([0, 100])
        .range([0, canvas_width - margin.left - margin.right])

      const scaleY = d3.scaleLinear().domain([0, 100])
        .range([canvas_height - margin.bottom - margin.top, 0])

      // Sublines.
      // Horizontal
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(
          d3.axisRight()
            .tickSize(canvas_width - margin.left - margin.right)
            .tickFormat('')
            .scale(scaleY)
        )

      // Vertical
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(
          d3.axisTop()
            .tickSize(-canvas_height + margin.top + margin.bottom)
            .tickFormat('')
            .scale(scaleX)
        )

      if (!this.tooltip) {
        this.tooltip = d3.select('#scatter-canvas')
          .append('div')
          .style('display', 'none')
          .style('position', 'absolute')
      }

      var ttip = this.tooltip

      const axX = d3.axisBottom(scaleX).ticks(5)
      const axY = d3.axisLeft(scaleY).ticks(5)

      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, canvas_height - margin.bottom] + ')')
        .attr('class', 'axis')
        .call(axX)
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(axY)

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
        .on('mouseenter', (m, index) => {
          let x = d3.event.layerX + 10
          let y = d3.event.layerY + 10
          if (x >= canvas_width * 0.8) {
            x -= 100
          }
          if (y >= canvas_height * 0.8) {
            y -= 80
          }

          const metric1 = m.getResultOfMetric1()
          const metric2 = m.getResultOfMetric2()
          ttip.style('display', 'inline-block')
          ttip.transition()
            .duration(200)
            .style('opacity', 0.9)
          ttip.html(
            'ID : ' + m.id + '<br />' +
            'Model: ' + this.getAlgorithmTitleFromId(m.algorithm_id) + '<br />' +
            metric1.metric + ' : ' + metric1.value + '<br />' +
            metric2.metric + ' : ' + metric2.value
          )
            .style('top', y + 'px')
            .style('left', x + 'px')
            .style('padding', '10px')
            .style('background', this.getAlgorithmColor(m.algorithm_id))
            .style('color', 'white')
            .style('text-align', 'left')
            .style('font-size', '0.8rem')
            .style('line-height', '1.1rem')
            .on('mouseleave', () => {
              ttip.style('display', 'none')
            })
        })
        .on('mouseleave', () => {
          ttip.style('display', 'none')
        })
        .on('click', (m) => {
          this.setSelectedModel(m)
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
    top: calc(100% - #{$scatter-padding}*1.5);
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding});
    height: $scatter-padding;
    text-align: center;
    font-size: $component-font-size-small;
    color: $component-font-color-title;
  }
  #title-metric2 {
    position: absolute;
    top: 0;
    left: calc(#{$scatter-padding}*0.7);
    width: $scatter-padding;
    height: 100%;
    writing-mode: vertical-rl;
    text-align: center;
    font-size: $component-font-size-small;
    color: $component-font-color-title;
  }
  #scatter-canvas {
    position: absolute;
    top: $scatter-padding;
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding}*2);
    height: calc(100% - #{$scatter-padding}*2);
    .axis {
      path {
        stroke: lightgray;
      }
      line {
        stroke: $scatter-grid-color;
      }
    }
    .tick {
      text {
        fill: $component-font-color-title;
      }
      line {
        stroke-dasharray: 2, 2;
      }
    }
  }
}
</style>
