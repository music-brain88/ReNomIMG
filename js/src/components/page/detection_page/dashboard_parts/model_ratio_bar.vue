<template>
  <div id="model-ratio-bar">
    <div class="row">
      <div class="col-md-3">
        <div class='title'>Total Models: {{$store.state.models.filter(model => model.state !== 3).length}}</div>
      </div>
      <div class="col-md-9 legend">
        <div class="row">
          <div class="col-md-2">
            <div class="nightblue">
              <span class="box"></span>YOLO
            </div>
          </div>
          <div class="col-md-2">
            <div class="lightblue">
              <span class="box"></span>YOLO2
            </div>
          </div>
          <div class="col-md-2">
            <div class="green">
              <span class="box"></span>SSD
            </div>
          </div>
          <div class="col-md-2">
            <div class="lightgreen">
              <span class="box"></span>DSSD
            </div>
          </div>
          <div class="col-md-2">
            <div class="yellow">
              <span class="box"></span>SPPnet
            </div>
          </div>
          <div class="col-md-2">
            <div class="grape">
              <span class="box"></span>R-FCN
            </div>
          </div>
        </div>
        <div class="row status">
          <div class="status-info">
            <div class="status-running">
              <span class="box"></span>Running
            </div>
          </div>
          <div class="status-info">
            <div class="status-reserve">
              <span class="box"></span>Reserved
            </div>
          </div>
        </div>
      </div>
    </div>
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
            categoryPercentage: 0.3
          }]
        },
        responsive: false,
        maintainAspectRatio: false,
        legend: {
          display: false,
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

  overflow: visible;

  .title {
    font-family:$content-inner-header-font-family;
    font-size: $content-inner-header-font-size;
  }
  .status{
    float: right;
    .status-info{
      display: flex;
      margin-left: 4px;
    }
  }
  .box {
    width: $content-figure-font-size;
    height: $content-figure-font-size;
    margin-right: 4px;
  }
  .grape, .nightblue, .lightblue, .green, .lightgreen,
  .yellow, .status-running, .status-reserve{
    display: inline-flex;
    font-size: $content-figure-font-size;
    vertical-align: baseline;
  }

  .grape .box{
    background-color: $grape;
  }

  .nightblue .box{
    background-color: $nightblue;
  }
  .lightblue .box{
    background-color: $lightblue;
  }
  .green .box{
    background-color: $green
  }
  .lightgreen .box{
    background-color: $lightgreen;
  }
  .yellow .box{
    background: $yellow;
  }
  .status-running .box{
    background-color: $status-running;
  }
  .status-reserve .box{
    background: $status-reserve;
  }
}
</style>
