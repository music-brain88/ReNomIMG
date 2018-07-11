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
// import Chart from 'chart.js'

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
      let curve_area = document.getElementById('learning-curve')

      let margin = { 'top': 30, 'bottom': 60, 'right': 30, 'left': 60 }

      let width = curve_area.clientWidth
      let height = curve_area.clientHeight

      var svg = d3.select('#curve-canvas').append('svg').attr('width', width).attr('height', height)

      let xScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d, index) { return index })])
        .range([margin.left, width - margin.right])

      let yScale = d3.scaleLinear()
        .domain([0, d3.max(datasets[0].data, function (d) { return d })])
        .range([height - margin.bottom, margin.top])

      // 4. 軸の表示
      let axisx = d3.axisBottom(xScale).ticks(5)
      let axisy = d3.axisLeft(yScale).ticks(5)
      svg.append('g')
        .attr('transform', 'translate(' + 0 + ',' + (height - margin.bottom) + ')')
        .call(axisx)
        .append('text')
        .attr('fill', 'black')
        .attr('x', (width - margin.left - margin.right) / 2 + margin.left)
        .attr('y', 35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10pt')
        .attr('font-weight', 'bold')
        .text('Epoch')

      svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + 0 + ')')
        .call(axisy)
        .append('text')
        .attr('fill', 'black')
        .attr('text-anchor', 'middle')
        .attr('x', -(height - margin.top - margin.bottom) / 2 - margin.top)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .attr('font-weight', 'bold')
        .attr('font-size', '10pt')
        .text('Loss')

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
      // let curve_area = document.getElementById('learning-curve')
      //
      // let canvas = document.getElementById('curve-canvas')
      // let ctx = canvas.getContext('2d')
      // ctx.canvas.width = curve_area.clientWidth
      // ctx.canvas.height = curve_area.clientHeight - 24 // minus title height
      //
      // const colors = ['#0762ad', '#ef8200']
      //
      // let datasets = [{
      //   label: 'train',
      //   fill: false,
      //   lineTension: 0,
      //   borderWidth: 3,
      //   borderColor: colors[0],
      //   backgroundColor: colors[0],
      //   pointRadius: 0.3,
      //   data: this.trainLoss
      // }, {
      //   label: 'validation',
      //   fill: false,
      //   lineTension: 0,
      //   borderWidth: 3,
      //   borderColor: colors[1],
      //   backgroundColor: colors[1],
      //   pointRadius: 0.3,
      //   data: this.validationLoss
      // }]
      //
      // // create x axis labels
      // let labels = []
      // let q = Math.ceil(this.trainLoss.length / 50)
      // if (this.trainLoss.length < 10) {
      //   labels = Array.from(new Array(10)).map((v, i) => i)
      // } else {
      //   labels = Array.from(new Array(q * 50)).map((v, i) => i)
      // }
      //
      // let data = {
      //   labels: labels,
      //   datasets: datasets
      // }
      // let options = {
      //   animation: {
      //     duration: 0
      //   },
      //   hover: {
      //     animationDuration: 0
      //   },
      //   layout: {
      //     padding: {
      //       top: 16,
      //       bottom: 0,
      //       left: 16,
      //       right: 16
      //     }
      //   },
      //   scales: {
      //     xAxes: [{
      //       ticks: {
      //         padding: -4,
      //         maxRotation: 0,
      //         minRotation: 0,
      //         autoSkip: true
      //       },
      //       gridLines: {
      //         color: 'rgba(0,0,0,0.1)',
      //         borderDash: [1, 1, 1]
      //       },
      //       scaleLabel: {
      //         display: true,
      //         labelString: 'Epoch',
      //         padding: {
      //           top: -4
      //         }
      //       }
      //     }],
      //     yAxes: [{
      //       ticks: {
      //         min: 0
      //       },
      //       gridLines: {
      //         color: 'rgba(0,0,0,0.1)',
      //         borderDash: [1, 1, 1]
      //       },
      //       scaleLabel: {
      //         display: true,
      //         labelString: 'Loss',
      //         padding: {
      //           bottom: 4
      //         }
      //       }
      //     }]
      //   },
      //   legend: {
      //     display: false
      //   },
      //   responsive: false,
      //   maintainAspectRatio: false
      // }
      //
      // if (this.learning_curve_chart) this.learning_curve_chart.destroy()
      // this.learning_curve_chart = new Chart(ctx, {
      //   type: 'line',
      //   data: data,
      //   options: options
      // })
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
  .axis text{
    font: 10px sans-serif;
  }
  .axis path {
  /*
  fillとopacityで半透明な背景色。
  */
    fill: Aquamarine;
    opacity: 0.1;
    stroke: #000;
    stroke-width: 1;
    shape-rendering: crispEdges;
  }

  .axis line {
    stroke: #000;
    stroke-width: 1;
    shape-rendering: crispEdges;
  }

  .line {
    fill: none;
    stroke: DarkGreen;
    stroke-width: 1.5px;
  }
}
</style>
