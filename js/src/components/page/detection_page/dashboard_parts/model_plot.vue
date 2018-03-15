<template>
  <div id="model-plot">
    <canvas id="accuracy-plot"></canvas>
  </div>
</template>

<script>
import { mapState } from 'vuex'
import Chart from 'chart.js'
import * as utils from '@/utils'
import * as constant from '@/constant'

export default {
  name: "ModelPlot",
  computed: mapState(["selected_model_id", "models"]),
  watch: {
    models() {
      this.drawScatter();
    },
    selected_model_id() {
      this.drawScatter();
    }
  },
  mounted: function() {
    this.drawScatter();
  },
  methods: {
    plotData: function(model_id, algorithm, best_iou, best_map) {
      return {
        "model_id": model_id,
        "algorithm": algorithm,
        "algorithm_name": constant.ALGORITHM_NAME[algorithm],
        "iou_value": utils.round_percent(best_iou) +"%",
        "map_value": utils.round_percent(best_map) +"%",
        "x": best_iou*100,
        "y": best_map*100
      }
    },
    drawScatter: function(models) {
      const self = this;

      // calc plot dataset
      const selected_color = "#999999"
      let tooltip_colors = [selected_color, constant.STATE_COLOR["Running"]];

      // init coordinate data
      let coordinate_data = {
        "Selected": {
          data:[],
          backgroundColor: selected_color,
          pointRadius: 10,
          pointHoverRadius: 12,
          pointStyle: "rectRot",
        },
        "Running": {
          data:[],
          backgroundColor: constant.STATE_COLOR["Running"],
          pointRadius: 4,
          pointHoverRadius: 6,
          pointStyle: "circle",
        }
      }
      // add initial coordinate per algorithm
      for(let k in Object.keys(constant.ALGORITHM_NAME)) {
        tooltip_colors.push(constant.ALGORITHM_COLOR[k]);

        coordinate_data[constant.ALGORITHM_NAME[k]] = {
          data:[],
          backgroundColor: constant.ALGORITHM_COLOR[k],
          pointRadius: 4,
          pointHoverRadius: 6,
          pointStyle: "circle",
        };
      }

      // add model coordinate to coordinate_data
      for(let model of this.models) {
        if(model.model_id == this.$store.state.selected_model_id) {
          coordinate_data["Selected"].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
        }else if(model.state == constant.ALGORITHM_ID["Running"]){
          coordinate_data["Running"].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
        }else{
          for(let k in Object.keys(constant.ALGORITHM_NAME)) {
            if(model.algorithm == k) {
              coordinate_data[constant.ALGORITHM_NAME[k]].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
            }
          }
        }
      }

      // create datasets from coordinate data
      let datasets = [coordinate_data["Selected"], coordinate_data["Running"]];
      for(let k in Object.keys(constant.ALGORITHM_NAME)) {
        datasets.push(coordinate_data[constant.ALGORITHM_NAME[k]]);
      }

      // canvas
      let parent = document.getElementById("model-plot")
      let parent_width = parent.style.width
      let parent_height = parent.style.height
      let canvas = document.getElementById("accuracy-plot");
      let ctx = canvas.getContext('2d');

      canvas.width = parent_width
      canvas.height = parent_height

      // set chart data
      let chart_data = {
        datasets: datasets,
      };

      // set chart options
      var options = {
        events: ['click', 'mousemove'],
        'onClick': function (evt, item) {
          console.log(item);
          if(item.length > 0)
          {
            const index = item[0]._index
            const datasetIndex = item[0]._datasetIndex
            const selectedModel = datasets[datasetIndex].data[index]
            const model_id = selectedModel.model_id
            self.$store.commit("setSelectedModel", {
              "model_id": model_id
            });
          }
        },
        animation: {
          duration: 0,
        },
        scales: {
          xAxes: [{
            position: 'bottom',
            ticks: {
              min: 0,
              max: 100,
            },
            scaleLabel:{
              display: true,
              labelString: 'IOU [%]',
            }
          }],
          yAxes: [{
            position: 'left',
            ticks: {
              min: 0,
              max: 100,
            },
            scaleLabel:{
              display: true,
              labelString: 'mAP [%]',
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
          // set custom tooltips
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
            // set custom tooltips
            title: function(item, data) {
              const model_id = datasets[item[0].datasetIndex].data[item[0].index].model_id;
              return "Model ID: " + model_id;
            },
            label: function(item, data) {
              const model = datasets[item.datasetIndex].data[item.index];
              const algorithm = model.algorithm_name;
              const iou_value = model.iou_value;
              const map_value = model.map_value;
              return ["Algorithm: "+algorithm, "IoU: "+iou_value, "mAP: "+map_value];
            },
            labelColor: function(item, chart) {
              chart.tooltip._model.backgroundColor = tooltip_colors[item.datasetIndex];
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

