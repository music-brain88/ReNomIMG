<template>
  <div id="model-ratio-bar">
    <div class='title'>Total Models: {{$store.state.models.length}}</div>
    <canvas id="horizontal-stack-bar"></canvas>
  </div>
</template>

<script>
import { mapState } from 'vuex'
import Chart from 'chart.js'
import * as constant from '@/constant'

export default {
  name: 'ModelRatioBar',
  computed: mapState(['models']),
  mounted: function () {
    this.drawBar()
  },
  updated: function () {
    this.drawBar()
  },
  methods: {
    drawBar: function () {
      // init counts of running & algorithm
      let counts = {'Running': 0}
      for (let k in Object.keys(constant.ALGORITHM_NAME)) {
        counts[constant.ALGORITHM_NAME[k]] = 0
      }

      // calc model counts per algorithm&running.
      for (let index in this.models) {
        if (this.models[index].state === 1) {
          counts['Running'] += 1
        } else if (this.models[index].algorithm === 0) {
          for (let k in Object.keys(constant.ALGORITHM_NAME)) {
            if (this.models[index].algorithm === k) {
              counts[constant.ALGORITHM_NAME[k]] += 1
            }
          }
        }
      }

      // crate chart dataset
      let datasets = []

      // add finished counts per algorithm
      for (let k in Object.keys(constant.ALGORITHM_NAME)) {
        datasets.push({
          label: constant.ALGORITHM_NAME[k],
          data: [this.models.filter(model => model.state === constant.STATE_ID['Finished'] && model.algorithm === parseInt(k)).length],
          backgroundColor: constant.ALGORITHM_COLOR[k]
        })
      }
      // add Running and Reserved counts
      for (let s of ['Running', 'Reserved']) {
        datasets.push({
          label: s,
          data: [this.models.filter(model => model.state === constant.STATE_ID[s]).length],
          backgroundColor: constant.STATE_COLOR[s]
        })
      }
      // init canvas
      let parent = document.getElementById('model-ratio-bar')
      let canvas = document.getElementById('horizontal-stack-bar')
      let ctx = canvas.getContext('2d')
      ctx.canvas.width = parent.clientWidth
      ctx.canvas.height = 100

      // set chart data
      let chart_data = {
        labels: [''],
        datasets: datasets
      }

      // set char options
      var options = {
        animation: {
          duration: 0
        },
        scales: {
          xAxes: [{
            display: false,
            stacked: true
          }],
          yAxes: [{
            display: false,
            stacked: true,
            barParcentage: 0.5,
            categoryPercentage: 0.5
          }]
        },
        responsive: false,
        maintainAspectRatio: false,
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            boxWidth: 10,
            fontSize: 10
          }
        },
        tooltips: {
          bodyFontSize: 10
        }
      }

      // remove chart for redraw.
      if (this.chart) this.chart.destroy()

      // draw chart
      this.chart = new Chart(ctx, {
        type: 'horizontalBar',
        data: chart_data,
        options: options })
    }
  }
}
</script>

<style lang="scss" scoped>
#model-ratio-bar {
  $title-height: 24px;
  $title-font-size: 16px;
  $font-weight-medium: 500;

  overflow: visible;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }
}
</style>

