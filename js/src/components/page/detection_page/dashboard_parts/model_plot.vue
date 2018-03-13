<template>
  <div id="model-plot">
    <canvas id="accuracy-plot"></canvas>
  </div>
</template>

<script>
import Chart from 'chart.js'

export default {
  name: "ModelPlot",
  computed: {
    plotDataset() {
      return this.$store.getters.getPlotDataset;
    },
  },
  watch: {
    plotDataset(newVal) {
      this.drawScatter(newVal);
    }
  },
  mounted: function() {
    this.drawScatter(this.plotDataset);
  },
  methods: {
    drawScatter: function(plotDataset) {
      let parent = document.getElementById("model-plot")
      let parent_width = parent.style.width
      let parent_height = parent.style.height
      let canvas = document.getElementById("accuracy-plot");
      let ctx = canvas.getContext('2d');

      canvas.width = parent_width
      canvas.height = parent_height

      let chart_data = {
        datasets: plotDataset.dataset,
        store: this.$store,
      };

      var options = {
        events: ['click', 'mousemove'],
        'onClick': function (evt, item) {
          if(item.length > 0)
          {
            let index = item[0]._index
            let datasetIndex = item[0]._datasetIndex
            let selectedModel = this.data.datasets[datasetIndex].data[index]
            let model_id = selectedModel.model_id
            let store = this.data.store

            store.commit("setSelectedModel", {
              "model_id": model_id
            });
          }
        },
        animation: {
          duration: 0,
        },
        scales: {
          xAxes: [{
            type: 'linear',
            position: 'bottom',
            ticks: {
              min: 0,
              max: 100,
              includeZero: true,
            },
            scaleLabel:{
              display: true,
              labelString: 'IOU [%]',
              padding: {
                top: -4,
              }
            }
          }],
          yAxes: [{
            type: 'linear',
            position: 'left',
            includeZero: true,
            ticks: {
              min: 0,
              max: 100,
            },
            scaleLabel:{
              display: true,
              labelString: 'mAP [%]',
              padding: {
                bottom: -8,
              }
            }
          }]
        },
        layout: {
          padding: {
            top: 24,
            bottom: 0,
            left: 24,
            right: 24,
          }
        },
        legend: false,
        tooltips: {
          enable: true,
          intersect: true,
          custom: function(tooltip) {
            if(!tooltip) return;
            tooltip.displayColors = false;
          },
          titleFontSize: 16,
          titleSpacing: 4,
          bodyFontSize: 14,
          xPadding: 12,
          yPadding: 12,
          callbacks: {
            title: function(item, data) {
              const model_id = data.datasets[item[0].datasetIndex].data[item[0].index].model_id;
              return "Model ID: " + model_id;
            },
            label: function(item, data) {
              const model = data.datasets[item.datasetIndex].data[item.index];
              const algorithm = model.algorithm_name;
              const iou_value = model.iou_value;
              const map_value = model.map_value;
              return ["Algorithm: "+algorithm, "IoU: "+iou_value, "mAP: "+map_value];
            },
            labelColor: function(item, chart) {
              chart.tooltip._model.backgroundColor = plotDataset.colors[item.datasetIndex];
            }
          }
        },
      };

      if(this.chart) this.chart.destroy();
      this.chart = new Chart(ctx, {
          type: 'scatter',
          data: chart_data,
          options: options });
    }
  }
}
</script>

<style lang="scss" scoped>

#model-plot {
  width: 100%;
  height: 100%;
}


</style>

