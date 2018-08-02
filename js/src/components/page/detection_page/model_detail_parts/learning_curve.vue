<template>
  <div id="learning-curve">
    <div class="curve-area">
      <div id='curve-canvas'></div>
      <div class="curve-legend">
        <div class="train">
          <span class="box"></span>Train
        </div>

        <div class="validation">
          <span class="box"></span>Validation
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as d3 from 'd3'

export default {
  name: 'LearningCurve',
  props: {
    'trainLoss': {
      type: Array,
      required: true
    },
    'validationLoss': {
      type: Array,
      required: true
    }
  },
  mounted: function () {
    this.drawLearningCurve()
  },
  watch: {
    trainLoss: function () {
      this.drawLearningCurve()
    }
  },
  methods: {
    drawLearningCurve: function () {
      const colors = ['#0762ad', '#ef8200']

      let datasets = [{
        label: 'train',
        Color: colors[0],
        data: this.trainLoss
      }, {
        label: 'validation',
        Color: colors[1],
        data: this.validationLoss
      }]

      this.removeData()

      let curve_area = document.getElementById('learning-curve')

      const margin = { 'top': 20, 'bottom': 60, 'right': 30, 'left': 60 }
      const width = curve_area.clientWidth
      const height = curve_area.clientHeight
      const inner_width = width - (margin.right + margin.left)
      const inner_height = height - (margin.top + margin.bottom)

      const zoom = d3.zoom()
        .scaleExtent([0, 10])
        .translateExtent([
          [0, 0],
          [width, height]
        ])
        .on('zoom', zoomed)

      const svg = d3.select('#curve-canvas')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .call(zoom)

      // Clipping SVG if Over axes
      svg.append('defs')
        .append('clipPath')
        .attr('id', 'clip')
        .append('rect')
        .attr('x', margin.left)
        .attr('y', margin.top)
        .attr('width', inner_width)
        .attr('height', inner_height)

      // resetZoom action
      d3.select('#curve-canvas').on('contextmenu', resetZoom)

      const xScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d, index) { return index })])
        .range([margin.left, width - margin.right])

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d) { return d })])
        .range([height - margin.bottom, margin.top])

      // get axes
      const axisx = d3.axisBottom(xScale)
        .tickSizeInner(-(height - (margin.top + margin.bottom)))
        .tickSizeOuter(0)
        .ticks(10)
        .tickPadding(10)

      const axisy = d3.axisLeft(yScale)
        .ticks(10)
        .tickSizeInner(-(width - (margin.left + margin.right)))
        .tickSizeOuter(0)
        .ticks(10)
        .tickPadding(10)

      // draw x axis
      let gX = svg.append('g')
        .attr('transform', 'translate(' + 0 + ',' + (height - margin.bottom) + ')')
        .call(axisx)

      // draw y axis
      let gY = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + 0 + ')')
        .call(axisy)

      stylingAxes()
      gX.append('text')
        .attr('fill', d3.rgb(0, 0, 0, 0.5))
        .attr('x', (width - margin.left - margin.right) / 2 + margin.left)
        .attr('y', 35)
        .style('font-size', '0.8em')
        .text('Epoch')

      gY.append('text')
        .attr('fill', d3.rgb(0, 0, 0, 0.5))
        .attr('x', -(height - margin.top - margin.bottom) / 2 - margin.top)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .style('font-size', '0.8em')
        .text('Loss')

      // Define tooltips
      let tooltips = d3.select('#learning-curve')
        .append('div')
        .style('display', 'none')

      let LineLayer = svg.append('g').attr('clip-path', 'url(#clip)')

      // draw line chart
      let TrainLine = LineLayer.append('path')
        .datum(datasets[0].data)
        .attr('fill', 'none')
        .attr('stroke', colors[0])
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return xScale(index) })
          .y(function (d) { return yScale(d) })
          .curve(d3.curveLinear)
        )

      // draw line chart
      let ValidationLine = LineLayer.append('path')
        .datum(datasets[1].data)
        .attr('fill', 'none')
        .attr('stroke', colors[1])
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return xScale(index) })
          .y(function (d) { return yScale(d) })
          .curve(d3.curveLinear)
        )

      let scatterLayer = svg.append('g').attr('clip-path', 'url(#clip)')

      // draw Train scatter
      let TrainScatter = scatterLayer.append('g')
        .selectAll('circle')
        .data(datasets[0].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[0])
        .attr('r', 2)
        .on('mousemove', function (d, index) {
          tooltips.style('display', 'inline-block')
          tooltips.html(index + '<br />' + 'Train:' + d)
            .style('position', 'relative')
            .style('cursor', 'default')
            .style('left', (d3.select(this).attr('cx') - 30) + 'px')
            .style('top', (d3.select(this).attr('cy') - height) + 'px')
            .style('color', d3.rgb(255, 255, 255, 0.8))
            .style('background', d3.rgb(0, 0, 0, 0.8))
            .style('padding', 2 + '%')
            .style('border-radius', 6 + 'px')
            .style('z-index', 10000)
            .on('mouseenter', function () {
              tooltips.style('display', 'inline-block')
            })
            .on('mouseleave', function () {
              tooltips.style('display', 'none')
            })
          d3.select(this).attr('r', 3)
        })
        .on('mouseleave', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 2)
        })

      // draw Validation scatter data
      let ValidationScatter = scatterLayer.append('g')
        .selectAll('circle')
        .data(datasets[1].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[1])
        .attr('r', 2)
        .on('mouseenter', function (d, index) {
          tooltips.style('display', 'inline-block')
          tooltips.html(index + '<br />' + 'Validation:' + d)
            .style('position', 'relative')
            .style('cursor', 'default')
            .style('left', (d3.select(this).attr('cx') - 30) + 'px')
            .style('top', (d3.select(this).attr('cy') - height) + 'px')
            .style('color', d3.rgb(255, 255, 255, 0.8))
            .style('background', d3.rgb(0, 0, 0, 0.8))
            .style('padding', 2 + '%')
            .style('border-radius', 6 + 'px')
            .style('z-index', 10000)
            .on('mouseenter', function () {
              tooltips.style('display', 'inline-block')
            })
            .on('mouseleave', function () {
              tooltips.style('display', 'none')
            })
          d3.select(this).attr('r', 3)
        })
        .on('mouseleave', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 2)
        })

      function zoomed () {
        gX.call(axisx.scale(d3.event.transform.rescaleX(xScale)))
          .selectAll('.tick:not(:first-child) line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')

        gY.call(axisy.scale(d3.event.transform.rescaleY(yScale)))
          .selectAll('line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')

        TrainLine.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        ValidationLine.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        TrainScatter.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        ValidationScatter.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        stylingAxes()
      }
      function stylingAxes () {
        gX.selectAll('path')
          .style('stroke', d3.rgb(128, 128, 128, 0.5))
        gX.selectAll('line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')
        gX.selectAll('.tick').selectAll('text')
          .style('fill', d3.rgb(0, 0, 0, 0.5))
          .style('font-size', '0.75em')

        gY.selectAll('path')
          .style('stroke', d3.rgb(128, 128, 128, 0.5))
        gY.selectAll('line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')
        gY.selectAll('.tick').selectAll('text')
          .style('fill', d3.rgb(0, 0, 0, 0.5))
          .style('font-size', '0.75em')
      }
      function resetZoom () {
        svg.transition()
          .duration(1000)
          .call(zoom.transform, d3.zoomIdentity)
        d3.event.preventDefault()
      }
    },
    removeData: function () {
      console.log('remove:', 'remove!!!!!')
      d3.select('#curve-canvas').selectAll('*')
        .remove()
    }
  }
}
</script>

<style lang="scss" scoped>
#learning-curve {
  $title-height: 24px;
  $title-font-size: 16px;
  $font-weight-medium: 500;

  $content-margin: 8px;

  $legend-font-size: 10px;
  $train-color: #0762ad;
  $validation-color: #ef8200;

  width: 100%;
  height: 100%;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .curve-area {
    position: relative;

    .curve-legend {
      position: absolute;
      top: 20px;
      right: 20px;

      .train, .validation {
        display: flex;
        font-size: $legend-font-size;
        line-height: $legend-font-size;
      }

      .validation {
        margin-top: 2px;
      }

      .box {
        width: $legend-font-size;
        height: $legend-font-size;
        margin-right: 4px;
      }

      .train .box {
        background-color: $train-color;
      }
      .validation .box {
        background-color: $validation-color;
      }
    }
  }
}
</style>
