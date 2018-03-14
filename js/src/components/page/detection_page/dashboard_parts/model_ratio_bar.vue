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
  name: "ModelRatioBar",
  computed: mapState(["models"]),
  watch: {
    models(newVal) {
      this.drawBar(newVal);
    }
  },
  mounted: function() {
    this.drawBar(this.models);
  },
  methods: {
    drawBar: function(models) {
      // calc model counts per algorithm&running.
      let counts = {"YOLO": 0, "SSD": 0, "Running": 0}
      for(let index in this.models) {
        if(this.models[index].state == 1){
          counts["Running"] += 1;
        }else if(this.models[index].algorithm == 0){
          counts["YOLO"] += 1;
        }
      }

      // init canvas
      let parent = document.getElementById("model-ratio-bar");
      let canvas = document.getElementById("horizontal-stack-bar");
      let ctx = canvas.getContext('2d');
      ctx.canvas.width = parent.clientWidth;
      ctx.canvas.height = 80;

      // set chart data
      let chart_data = {
        labels: [""],
        datasets: [{
          label: "YOLO",
          data: [counts["YOLO"]],
          backgroundColor: constant.ALGORITHM_COLOR["YOLO"],
        },
        {
          label: "Running",
          data: [counts["Running"]],
          backgroundColor: constant.STATE_COLOR["Running"],
        }
      ]};

      // set char options
      var options = {
        animation: {
          duration: 0,
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
            categoryPercentage: 0.5,
          }]
        },
        responsive: false,
        maintainAspectRatio: false,
        legend: {
          display:true,
          position: 'bottom',
          labels: {
            boxWidth: 10,
            fontSize: 10
          }
        }
      };

      // remove chart for redraw.
      if(this.chart) this.chart.destroy();

      // draw chart
      this.chart = new Chart(ctx, {
        type: 'horizontalBar',
        data: chart_data,
        options: options });
    }
  }
}
</script>

<style lang="scss" scoped>
#model-ratio-bar {
  $title-height: 24px;
  $title-font-size: 16px;
  $font-weight-medium: 500;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }
}
</style>

