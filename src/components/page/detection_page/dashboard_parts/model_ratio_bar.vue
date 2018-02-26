<template>
  <div id="model-ratio-bar">
    <div class='title'>Total Models: {{modelAllCounts}}</div>
    <canvas id="horizontal-stack-bar"></canvas>
  </div>
</template>

<script>
import Chart from 'chart.js'

export default {
  name: "ModelRatioBar",
  mounted: function() {
    if(this.modelCounts) {
      this.drawBar(this.modelCounts);
    }
  },
  watch: {
    modelCounts(newVal) {
      this.drawBar(newVal);
    }
  },
  computed: {
    modelCounts() {
      return this.$store.getters.getModelCounts;
    },
    modelAllCounts() {
      let d = this.$store.getters.getModelCounts;
      let count = 0
      for(let attr in d){
        count += d[attr]
      }
      return count
    },
    algorithmId() {
      return this.$store.state.const.algorithm_id;
    },
    algorithmColor() {
      return this.$store.state.const.algorithm_color;
    },
    stateId() {
      return this.$store.state.const.state_id;
    },
    stateColor() {
      return this.$store.state.const.state_color;
    }
  },
  methods: {
    drawBar: function(modelCounts) {
      let parent = document.getElementById("model-ratio-bar")
      let canvas = document.getElementById("horizontal-stack-bar");
      let ctx = canvas.getContext('2d');
      ctx.canvas.width = parent.clientWidth;
      ctx.canvas.height = 80;

      let chart_data = {
        labels: [""],
        datasets: [{
          label: "YOLO",
          data: [modelCounts["YOLO"]],
          backgroundColor: this.algorithmColor[this.algorithmId["YOLO"]],
        },
        {
          label: "SSD",
          data: [modelCounts["SSD"]],
          backgroundColor: this.algorithmColor[this.algorithmId["SSD"]],
        },
        {
          label: "Running",
          data: [modelCounts["Running"]],
          backgroundColor: this.stateColor[this.stateId["running"]],
        }
      ]};

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

      if(this.chart) this.chart.destroy();
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

