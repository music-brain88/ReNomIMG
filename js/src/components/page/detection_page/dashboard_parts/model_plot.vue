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
    models(newVal) {
      this.drawScatter(newVal);
    },
    selected_model_id() {
      this.drawScatter(this.models);
    }
  },
  mounted: function() {
    this.drawScatter(this.models);
  },
  methods: {
    plotData: function(model_id, algorithm, best_iou, best_map) {
      return {
        "model_id": model_id,
        "algorithm": algorithm,
        "algorithm_name": constant.ALGORITHM_NAME[algorithm],
        "iou_value": utils.round(best_iou, 100)*100 +"%",
        "map_value": utils.round(best_map, 100)*100 +"%",
        "x": best_iou*100,
        "y": best_map*100
      }
    },
    drawScatter: function(models) {
      // calc plot dataset
      let tooltip_colors = ["#999999", constant.STATE_COLOR["Running"], constant.ALGORITHM_COLOR["YOLO"]]
      let dataset = {
        "dataset": []
      };

      let coordinate_data = {
        "Selected": {
          data:[],
          backgroundColor: "#999999",
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
        },
        "YOLO": {
          data:[],
          backgroundColor: constant.ALGORITHM_COLOR["YOLO"],
          pointRadius: 4,
          pointHoverRadius: 6,
          pointStyle: "circle",
        }
      }

      for(let model of models) {
        if(model.model_id == this.$store.state.selected_model_id) {
          coordinate_data["Selected"].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
        }else if(model.state == constant.ALGORITHM_ID["Running"]){
          coordinate_data["Running"].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
        }else if(model.algorithm == 0){
          coordinate_data["YOLO"].data.push(this.plotData(model.model_id, model.algorithm, model.best_epoch_iou, model.best_epoch_map));
        }
      }

      dataset.dataset.push(coordinate_data["Selected"],coordinate_data["Running"],coordinate_data["YOLO"]);

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
        datasets: dataset.dataset,
        store: this.$store,
      };

      // set chart options
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
            // set custom tooltips
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

