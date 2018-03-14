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
  computed: {
    modelCounts() {
      return this.$store.getters.getModelCounts;
    },
    modelAllCounts() {
      if(this.$store.state.project){
        return this.$store.state.project.models.length;
      }
    }
  },
  watch: {
    modelCounts(newVal) {
      this.drawBar(newVal);
    }
  },
  methods: {
    drawBar: function(modelCounts) {
      const const_params = this.$store.state.const;

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
          backgroundColor: const_params.algorithm_color[const_params.algorithm_id["YOLO"]],
        },
        {
          label: "SSD",
          data: [modelCounts["SSD"]],
          backgroundColor: const_params.algorithm_color[const_params.algorithm_id["SSD"]],
        },
        {
          label: "Running",
          data: [modelCounts["Running"]],
          backgroundColor: const_params.state_color[1],
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

