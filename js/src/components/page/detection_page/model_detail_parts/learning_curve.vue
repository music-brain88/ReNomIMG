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

      // if (this.old_chart_data !== datasets) {
      //   this.removeData(this.old_chart_data[0].data)
      //   this.removeData(this.old_chart_data[1].data)
      // }

      let curve_area = document.getElementById('learning-curve')

      const margin = { 'top': 20, 'bottom': 60, 'right': 30, 'left': 60 }

      const width = curve_area.clientWidth
      const height = curve_area.clientHeight

      let svg = d3.select('#curve-canvas').append('svg').attr('width', width).attr('height', height)

      let xScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d, index) { return index })])
        .range([margin.left, width - margin.right])

      let yScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d) { return d })])
        .range([height - margin.bottom, margin.top])

      // get axes
      let axisx = d3.axisBottom(xScale)
        .tickSizeInner(-height)
        .tickSizeOuter(0)
        .ticks(10)
        .tickPadding(10)

      let axisy = d3.axisLeft(yScale)
        .ticks(10)
        .tickSizeInner(-width)
        .tickSizeOuter(0)
        .ticks(10)
        .tickPadding(10)

      // draw x axis
      svg.append('g')
        .attr('transform', 'translate(' + 0 + ',' + (height - margin.bottom) + ')')
        .call(axisx)
        .append('text')
        .attr('fill', d3.rgb(0, 0, 0, 0.5))
        .attr('x', (width - margin.left - margin.right) / 2 + margin.left)
        .attr('y', 35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '2pt')
        .attr('font-family', 'Inter UI', 'sans-serif')
        .text('Epoch')

      // draw y axis
      svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + 0 + ')')
        .call(axisy)
        .attr('class', 'axisRed')
        .append('text')
        .attr('fill', d3.rgb(0, 0, 0, 0.5))
        .attr('text-anchor', 'middle')
        .attr('x', -(height - margin.top - margin.bottom) / 2 - margin.top)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .attr('font-size', '2pt')
        .attr('font-family', 'Inter UI', 'sans-serif')
        .text('Loss')

      // Difine tooltips
      let tooltips = d3.select('#learning-curve')
        .append('div')
        .attr('class', 'tooltip')
        .style('display', 'none')

      // draw Train data dot
      svg.append('g')
        .selectAll('circle')
        .data(datasets[0].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[0])
        .attr('r', 3)
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
          d3.select(this).attr('r', 6)
        })
        .on('mouseout', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 3)
        })

      // draw Validation data dot
      svg.append('g')
        .selectAll('circle')
        .data(datasets[1].data)
        .enter()
        .append('circle')
        .attr('cx', function (d, index) { return xScale(index) })
        .attr('cy', function (d) { return yScale(d) })
        .attr('fill', colors[1])
        .attr('r', 3)
        .on('mouseover', function (d, index) {
          tooltips.style('display', 'inline-block')
          tooltips.html(index + '<br />' + 'Validation:' + d)
            .style('left', (d3.select(this).attr('cx') - 30) + 'px')
            .style('top', (d3.select(this).attr('cy') - height) + 'px')
            .style('color', d3.rgb(255, 255, 255, 0.8))
            .style('background', d3.rgb(0, 0, 0, 0.8))
            .style('padding', 0.2 + '%')
            .style('border-radius', 6 + 'px')
            .style('z-index', 10000)
          d3.select(this).attr('r', 6)
        })
        .on('mouseout', function (d) {
          tooltips.style('display', 'none')
          d3.select(this).attr('r', 3)
        })

      // draw line chart
      svg.append('path')
        .datum(datasets[0].data)
        .attr('fill', 'none')
        .attr('stroke', colors[0])
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return xScale(index) })
          .y(function (d) { return yScale(d) })
          .curve(d3.curveCardinal)
        )

      // draw line chart
      svg.append('path')
        .datum(datasets[1].data)
        .attr('fill', 'none')
        .attr('stroke', colors[1])
        .attr('stroke-width', 1.5)
        .attr('d', d3.line()
          .x(function (d, index) { return xScale(index) })
          .y(function (d) { return yScale(d) })
          .curve(d3.curveCardinal)
        )

      console.log('width', width)
      console.log('height', height)
      console.log('svg', svg)

      this.old_chart_data = datasets
    },
    removeData: function (dataset) {
      console.log('remove:', 'remove!!!!!')
      d3.select('#curve-canvas').selectAll('*')
        .data(dataset)
        .exit()
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
.tooltip {
  position: absolute;
  text-align: center;
  width: 60px;
  height: 28px;
}
</style>
