<template>
  <div id="learning-curve">
    <div class="title">Learning Curve</div>
    <div class="curve-area">
      <canvas id="curve-canvas"></canvas>
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
import Chart from 'chart.js'

export default {
  name: "LearningCurve",
  props: {
    "totalEpoch": {
      type: Number,
      required: true
    },
    "trainLoss": {
      type: Array,
      required: true
    },
    "validationLoss": {
      type: Array,
      required: true
    },
  },
  mounted: function() {
    this.drawLearningCurve();
  },
  watch: {
    trainLoss: function(newVal) {
      this.drawLearningCurve(newVal);
    }
  },
  methods: {
    drawLearningCurve: function() {
      let curve_area = document.getElementById("learning-curve");

      let canvas = document.getElementById("curve-canvas");
      let ctx = canvas.getContext('2d');
      ctx.canvas.width = curve_area.clientWidth;
      ctx.canvas.height = curve_area.clientHeight - 24; // minus title height

      const colors = ['#0762ad', '#ef8200'];

      let datasets = [{
        label: "train",
        fill: false,
        lineTension: 0,
        borderWidth: 3,
        borderColor: colors[0],
        backgroundColor: colors[0],
        pointRadius: 0.1,
        data: this.trainLoss,
      },{
        label: "validation",
        fill: false,
        lineTension: 0,
        borderWidth: 3,
        borderColor: colors[1],
        backgroundColor: colors[1],
        pointRadius: 0.1,
        data: this.validationLoss,
      }];

      let labels = Array.from(new Array(this.totalEpoch)).map((v,i) => i)

      let data = {
        labels: labels,
        datasets: datasets
      };
      let options = {
        animation: {
          duration: 0,
        },
        hover: {
          animationDuration: 0,
        },
        layout: {
          padding: {
            top: 16,
            bottom: 0,
            left: 16,
            right: 16
          }
        },
        scales: {
          xAxes: [{
            ticks: {
              padding: -4,
              maxRotation: 0,
              minRotation: 0,
              callback: function(value) {return ((value % 10) == 0)? value : ''},
            },
            gridLines: {
              color: "rgba(0,0,0,0.1)",
              borderDash: [1,1,1],
            },
            scaleLabel: {
              display: true,
              labelString: 'Epoch',
              padding: {
                top: -4,
              }
            }
          }],
          yAxes: [{
            ticks: {
              min: 0,
            },
            gridLines: {
              color: "rgba(0,0,0,0.1)",
              borderDash: [1,1,1],
            },
            scaleLabel: {
              display: true,
              labelString: 'Loss',
              padding: {
                bottom: 4,
              }
            }
          }]
        },
        legend: {
          display: false,
        },
        responsive: false,
        maintainAspectRatio: false
      }

      if(this.learning_curve_chart) this.learning_curve_chart.destroy();
      this.learning_curve_chart = new Chart(ctx, {
        type: 'line',
        data: data,
        options: options
      });
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
</style>

