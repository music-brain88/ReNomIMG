<template>
  <component-frame :width-weight="4" :height-weight="4">
    <template slot="header-slot">
      Learning Curve
    </template>
    <div id="learning-curve">
      <div id="title-epoch">
        Epoch [-]
      </div>
      <div id="title-loss">
        Loss [-]
      </div>
      <div id="learning-curve-canvas">
      </div>
    </div>
  </component-frame>
</template>

<script>
import * as d3 from 'd3'
import * as utils from '@/utils.js'
import { mapGetters } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentLearningCurve',
  components: {
    'component-frame': ComponentFrame
  },
  computed: {
    ...mapGetters(['getSelectedModel']),
    getTrainLossList: function () {
      const model = this.getSelectedModel
      if (model) {
        return model.train_loss_list
      } else {
        return []
      }
    }
  },
  created: function () {
    window.addEventListener('resize', this.draw, false)
  },
  mounted: function () {
    this.draw()
  },
  watch: {
    getTrainLossList: function () {
      // Watches the change of train_loss_list.
      this.draw()
    }
  },
  beforeDestroy: function () {
    // Remove it.
    window.removeEventListener('resize', this.draw, false)
  },
  methods: {
    draw: function () {
      d3.select('#learning-curve-canvas').select('svg').remove() // Remove SVG if it has been created.
      
      const margin = {top: 15, left: 40, right: 20, bottom: 20}
      const canvas = document.getElementById('learning-curve-canvas')
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const svg = d3.select('#learning-curve-canvas').append('svg').attr('id', 'learnning-graph')
      const train_color = '#0762ad'
      const valid_color = '#ef8200'
      let best_epoch = 0
      let train_loss_list = []
      let valid_loss_list = []

      if (this.getSelectedModel) {
        const model = this.getSelectedModel
        best_epoch = model.best_epoch_valid_result
        best_epoch = (best_epoch) ? best_epoch.nth_epoch : 0
        train_loss_list = model.train_loss_list
        valid_loss_list = model.valid_loss_list
      }
      const learning_epoch = train_loss_list.length
      let maxX = Math.max(learning_epoch + 1, 10)
      maxX = Math.ceil(maxX / 5) * 5
      let minX = 0
      let maxY = Math.max((Math.max.apply(null, [...train_loss_list, ...valid_loss_list]) * 1.1), 1)
      maxY = Math.ceil(maxY)
      let minY = Math.min(Math.min.apply(null, [...train_loss_list, ...valid_loss_list]), 0)
      minY = Math.floor(minY)
      console.log(minX, maxX)

      let tooltip = d3.select('#learning-curve-canvas')
        .append('div')
        .append('display', 'none')
      
      const zoom = d3.zoom()
        .scaleExtent([0, 10])
        .translateExtent([[0, 0], [canvas_width, canvas_height]])
        .on('zoom', zoomed)
      

      // Set size.
      svg.attr('width', canvas_width)
        .attr('height', canvas_height)
        .call(zoom)

      // Axis Settings
      const scaleX = d3.scaleLinear().domain([minX, maxX])
        .range([0, canvas_width - margin.left - margin.right])
      const scaleY = d3.scaleLinear().domain([minY, maxY])
        .range([canvas_height - margin.bottom - margin.top, 0])

      // Sublines.
      // Horizontal
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line')
        .call(
          d3.axisRight()
            .tickSize(canvas_width - margin.left - margin.right)
            .tickFormat('').ticks(5)
            .scale(scaleY)
        )

      // Vertical
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line')
        .call(
          d3.axisTop()
            .tickSize(-canvas_height + margin.top + margin.bottom)
            .tickFormat('').ticks(5)
            .scale(scaleX)
        )

      const axX = d3.axisBottom(scaleX).ticks(5)
      const axY = d3.axisLeft(scaleY).ticks(5)
      let gX = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, canvas_height - margin.bottom] + ')')
        .call(axX)
      let gY = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .call(axY)

      // Line graph
      let TrainLine = svg.append('path')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .datum(train_loss_list)
        .attr('fill', 'none')
        .attr('stroke', train_color)
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return scaleX(index) })
          .y(function (d) { return scaleY(d) })
          .curve(d3.curveLinear)
        )

      let ValidLine = svg.append('path')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .datum(valid_loss_list)
        .attr('fill', 'none')
        .attr('stroke', valid_color)
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return scaleX(index) })
          .y(function (d) { return scaleY(d) })
          .curve(d3.curveLinear)
        )

      let BestEpoc = svg.append('line')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 1.5)
        .attr('x1', scaleX(best_epoch))
        .attr('y1', scaleY(maxY))
        .attr('x2', scaleX(best_epoch))
        .attr('y2', scaleY(minY))

      // Scatter graph
      let TrainScatter = svg.append('g')
        .selectAll('circle')
        .data(train_loss_list)
        .enter()
        .append('circle')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('cx', function (d, index) {
          return scaleX(index)
        })
        .attr('cy', (d) => {
          return scaleY(d)
        })
        .attr('fill', train_color)
        .attr('r', 2)
        .on('mousemove', (d, index) => {
          // find svg id and get mouse position
          let x = d3.event.pageX - document.getElementById('learnning-graph')
            .getBoundingClientRect().x + 10
          let y = -(d3.event.pageY - document.getElementById('learnning-graph')
            .getBoundingClientRect().y)
          tooltip.style('display', 'inline-block')
          tooltip.transition()
            .duration(200)
            .style('opacity', 0.9)
          tooltip.html(
            'Epoc:' + (index + 1) + '<br />' +
            'Valid:' + utils.round(d, 1000) + '<br />' +
            'Y:' + y + '<br />' +
            'X:' + x + '<br />'
          )

            .style('position', 'absolute')
            .style('background', train_color)
            .style('top', y + 'px')
            .style('left', x + 'px')
        })
        .on('mouseleave', () => {
          tooltip.style('display', 'none')
        })

      let ValidScatter = svg.append('g')
        .selectAll('circle')
        .data(valid_loss_list)
        .enter()
        .append('circle')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('cx', function (d, index) {
          return scaleX(index)
        })
        .attr('cy', (d) => {
          return scaleY(d)
        })
        .attr('fill', valid_color)
        .attr('r', 2)
        .on('mousemove', (d, index) => {
          let x = d3.event.pageX - document.getElementById('learnning-graph')
            .getBoundingClientRect().x + 10
          let y = -(d3.event.pageY - document.getElementById('learnning-graph')
            .getBoundingClientRect().y + 10)
          tooltip.style('display', 'inline-block')
          tooltip.transition()
            .duration(200)
            .style('opacity', 0.9)
          tooltip.html(
            'Epoc:' + (index) + '<br />' +
            'Valid:' + utils.round(d, 1000) + '<br />' +
            'Y:' + y + '<br />' +
            'X:' + x + '<br />'
          )

            .style('position', 'relative')
            .style('background', valid_color)
            .style('top', y + 'px')
            .style('left', x + 'px')
        })
        .on('mouseleave', () => {
          tooltip.style('display', 'none')
        })

      d3.select('#learning-curve-canvas')
        .on('contextmenu', resetZoom)

      function zoomed () {
        TrainLine.attr('transform', 'translate(' + [margin.left, margin.top] + ') scale(' + d3.event.transform.k + ')')
        ValidLine.attr('transform', 'translate(' + [margin.left, margin.top] + ') scale(' + d3.event.transform.k + ')')
        TrainScatter.attr('transform', 'translate(' + [margin.left, margin.top] + ') scale(' + d3.event.transform.k + ')')
        ValidScatter.attr('transform', 'translate(' + [margin.left, margin.top] + ') scale(' + d3.event.transform.k + ')')
        BestEpoc.attr('transform', 'translate(' + [margin.left, margin.top] + ') scale(' + d3.event.transform.k + ')')
        gX.call(axX.scale(d3.event.transform.rescaleX(scaleX)))
        gY.call(axY.scale(d3.event.transform.rescaleY(scaleY)))
      }
      function resetZoom () {
        svg.call(zoom.transform, d3.zoomIdentity)
        TrainLine.attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        ValidLine.attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        TrainScatter.attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        ValidScatter.attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        d3.event.preventDefault()
      }
    }
  }
}
</script>

<style lang='scss'>
#learning-curve {
  width: 100%;
  height: 100%;
  padding: 20px;
  #learning-curve-canvas {
    width: 100%;
    height: 100%;
    .grid-line line {
      stroke: $scatter-grid-color;
    }
  }
}
</style>
    }
  }
}
</script>

<style lang='scss'>
#learning-curve {
  width: 100%;
  height: 100%;
  padding: $scatter-padding;
  position: relative;
  #title-epoch {
    position: absolute;
    top: calc(100% - #{$scatter-padding});
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding});
    height: $scatter-padding;
    text-align: center;
    font-size: 70%;
  }
  #title-loss {
    position: absolute;
    top: 0;
    left: calc(#{$scatter-padding}*0.7);
    width: $scatter-padding;
    height: 100%;
    writing-mode: vertical-rl;
    text-align: center;
    font-size: 70%;
  }

  #learning-curve-canvas {
    width: 100%;
    height: 100%;
    .grid-line line {
      stroke: $scatter-grid-color;
    }
  }
}
</style>
