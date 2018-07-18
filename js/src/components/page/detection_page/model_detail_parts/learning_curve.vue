<template>
  <div id="learning-curve">
    <div class="title">Learning Curve</div>
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
        fill: false,
        lineTension: 0,
        borderWidth: 3,
        borderColor: colors[0],
        backgroundColor: colors[0],
        pointRadius: 0.3,
        data: this.trainLoss
      }, {
        label: 'validation',
        fill: false,
        lineTension: 0,
        borderWidth: 3,
        borderColor: colors[1],
        backgroundColor: colors[1],
        pointRadius: 0.3,
        data: this.validationLoss
      }]

      this.removeData()

      // let dataset = d3.nest().key(function (d) { return d.label }).entries(datasets)

      // console.log('datasets', dataset)

      let curve_area = document.getElementById('learning-curve')

      const margin = { 'top': 20, 'bottom': 60, 'right': 30, 'left': 60 }
      const width = curve_area.clientWidth
      const height = curve_area.clientHeight

      const svg = d3.select('#curve-canvas').append('svg').attr('width', width).attr('height', height)

      // resetZoom action
      d3.select('#curve-canvas').on('contextmenu', resetZoom)

      const xScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d, index) { return index })])
        .range([margin.left, width - margin.right])

      // .range([margin.left, width - margin.right])

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d) { return d })])
        .range([height - margin.bottom, margin.top])
      // .range([height - margin.bottom, margin.top])

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
        .style('font-size', '0.85em')
        .text('Epoch')

      gY.append('text')
        .attr('fill', d3.rgb(0, 0, 0, 0.5))
        .attr('x', -(height - margin.top - margin.bottom) / 2 - margin.top)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .style('font-size', '0.85em')
        .text('Loss')

      // console.log(gY + gX)

      // Define tooltips
      let tooltips = d3.select('#learning-curve')
        .append('div')
        .attr('class', 'tooltip')
        .style('display', 'none')

      // draw line chart
      let TrainLine = svg.append('path')
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
      let ValidationLine = svg.append('path')
        .datum(datasets[1].data)
        .attr('fill', 'none')
        .attr('stroke', colors[1])
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return xScale(index) })
          .y(function (d) { return yScale(d) })
          .curve(d3.curveLinear)
        )

      // draw Train data dot
      let TrainDots = svg.append('g')
        .selectAll('circle')
        .data(datasets[0].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[0])
        .attr('r', 2)
        .on('mouseover', function (d, index) {
          tooltips.style('display', 'inline-block')
          tooltips.html(index + '<br />' + 'Train:' + d)
            .style('left', (d3.select(this).attr('cx') - 30) + 'px')
            .style('top', (d3.select(this).attr('cy') - height) + 'px')
            .style('color', d3.rgb(255, 255, 255, 0.8))
            .style('background', d3.rgb(0, 0, 0, 0.8))
            .style('padding', 2 + '%')
            .style('border-radius', 6 + 'px')
            .style('z-index', 10000)
            .on('mouseover', function (d, index) {
              tooltips.style('display', 'inline-block')
            })
            .on('mouseout', function (d) {
              tooltips.style('display', 'none')
              d3.select(this).attr('r', 2)
            })
          d3.select(this).attr('r', 3)
        })
        .on('mouseout', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 2)
        })

      // draw Validation data dot
      let ValidationDots = svg.append('g')
        .selectAll('circle')
        .data(datasets[1].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[1])
        .attr('r', 2)
        .on('mouseover', function (d, index) {
          tooltips.style('display', 'inline-block')
          tooltips.html(index + '<br />' + 'Validation:' + d)
            .style('left', (d3.select(this).attr('cx') - 30) + 'px')
            .style('top', (d3.select(this).attr('cy') - height) + 'px')
            .style('color', d3.rgb(255, 255, 255, 0.8))
            .style('background', d3.rgb(0, 0, 0, 0.8))
            .style('padding', 2 + '%')
            .style('border-radius', 6 + 'px')
            .style('z-index', 10000)
            .on('mouseover', function (d, index) {
              tooltips.style('display', 'inline-block')
            })
            .on('mouseout', function (d) {
              tooltips.style('display', 'none')
              d3.select(this).attr('r', 2)
            })
          d3.select(this).attr('r', 3)
        })
        .on('mouseout', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 2)
        })

      let zoom = d3.zoom()
        .scaleExtent([0, 10])
        .translateExtent([
          [0, 0],
          [width, height]
        ])
        .on('zoom', zoomed)

      svg.call(zoom)

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
        TrainDots.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        ValidationDots.attr('transform', 'translate(' + d3.event.transform.x + ',' + d3.event.transform.y + ') scale(' + d3.event.transform.k + ')')
        stylingAxes()
      }
      function stylingAxes () {
        gX.selectAll('path')
          .style('stroke', d3.rgb(128, 128, 128, 0.5))
        gX.selectAll('line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')
        gX.selectAll('text')
          .style('fill', d3.rgb(0, 0, 0, 0.5))
          .style('font-size', '0.85em')

        gY.selectAll('path')
          .style('stroke', d3.rgb(128, 128, 128, 0.5))
        gY.selectAll('line')
          .style('stroke', d3.rgb(0, 0, 0, 0.2))
          .style('stroke-dasharray', '2,2')
        gY.selectAll('text')
          .style('fill', d3.rgb(0, 0, 0, 0.5))
          .style('font-size', '0.85em')
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

  svg {
    flex-grow: 0;
    flex-shrink: 0;
    margin-top: 1rem;
  }

  .tooltip {
    content: '';
    position: absolute;
    text-align: center;
    width: 60px;
    height: 28px;
  }

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
