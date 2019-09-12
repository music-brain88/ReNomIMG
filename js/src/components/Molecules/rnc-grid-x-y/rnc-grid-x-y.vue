<template>
  <div
    id="learning-curve"
    @mouseenter="TooltipDisplay = false"
  >
    <div id="title-epoch">
      {{ axisNameXData }}
    </div>
    <div id="title-loss">
      {{ axisNameYData }}
    </div>
    <div
      ref="canvas"
      :id="idKind"
    >
      <rnc-tooltip
        :top="TooltipTop"
        :left="TooltipLeft"
        :display="TooltipDisplay"
        :text-array="TooltipTextArray"
        :kind="TooltipKind"
      />
    </div>
  </div>
</template>

<script>
import * as d3 from 'd3'
import { train_color, valid_color } from './../../../const_style'
import { STATE } from './../../../const'
import RncTooltip from './../../Atoms/rnc-tooltip/rnc-tooltip.vue'
import { getAlgorithmColor } from './../../../utils.js'

export default {
  name: 'RncGridXY',
  components: {
    'rnc-tooltip': RncTooltip
  },
  props: {
    kind: {
      type: String,
      default: 'learning-curve',
      validator: val => ['learning-curve', 'model-scatter'].includes(val)
    },
    axisNameX: {
      type: String,
      default: ''
    },
    axisNameY: {
      type: String,
      default: ''
    },
    selectedModelObj: {
      type: Object,
      default: function () { return undefined }
    },
    filteredAndGroupedModelListArray: {
      type: Array,
      default: function () { return undefined }
    },
    switchTrainGraph: {
      type: Boolean,
      default: false
    },
    switchValidGraph: {
      type: Boolean,
      default: false
    },
    algorithmTitleFunc: {
      type: Function,
      default: function () { return undefined }
    },
    percentMagnification: {
      type: Boolean,
      default: false
    },
    endOfAxisXY: {
      type: Object,
      default: function () {
        return {
          'x': {
            'max': 100,
            'min': 0
          },
          'y': {
            'max': 100,
            'min': 0
          }
        }
      }
    }
  },
  data: function () {
    return {
      train_graph_flg: true,
      valid_graph_flg: true,
      TooltipTop: 0,
      TooltipLeft: 0,
      TooltipDisplay: false,
      TooltipTextArray: undefined,
      TooltipKind: 'no-model',
      axisNameXData: '',
      axisNameYData: '',

      margin: { top: 15, left: 40, right: 20, bottom: 35 },
      TrainLine: undefined,
      ValidLine: undefined,
      TrainScatter: undefined,
      ValidScatter: undefined,
      BestEpoc: undefined,
      gX: undefined,
      gY: undefined,
      axX: undefined,
      axY: undefined,
      scaleX: undefined,
      scaleY: undefined,
      canvas_width: undefined,
      canvas_height: undefined,
      PlotModel: undefined
    }
  },
  computed: {
    getTrainLossList: function () {
      const model = this.selectedModelObj
      if (undefined !== model) {
        return model.train_loss_list
      } else {
        return []
      }
    },
    idKind () {
      return `${this.kind}-canvas`
    },
    zoom: function () {
      const zoom = d3.zoom()
        .scaleExtent([1, 3])
        .translateExtent([[0, 0], [this.canvas_width, this.canvas_height]])
        .on('zoom', this.zoomed, { passive: true })
      return zoom
    }
  },
  watch: {
    getTrainLossList: function () {
      // Watches the change of train_loss_list.
      this.drawLearningCurve()
    },
    filteredAndGroupedModelListArray: function () {
      this.drawModelScatter()
    },
    switchTrainGraph: function () {
      if (this.switchTrainGraph === false) {
        this.drawLearningCurve()
      } else {
        d3.select('#train-line').remove()
        d3.select('#train-scatter').remove()
      }
    },
    switchValidGraph: function () {
      if (this.switchValidGraph === false) {
        this.drawLearningCurve()
      } else {
        d3.select('#valid-line').remove()
        d3.select('#valid-scatter').remove()
      }
    },
    selectedModelObj: function () {
      this.drawModelScatter()
    }
  },
  mounted: function () {
    if (this.kind === 'learning-curve') {
      this.axisNameXData = this.axisNameX
      this.axisNameYData = this.axisNameY
      this.drawLearningCurve()
      window.addEventListener('resize', this.drawLearningCurve, false)
    } else if (this.kind === 'model-scatter') {
      this.drawModelScatter()
      window.addEventListener('resize', this.drawModelScatter, false)
    }
  },
  beforeDestroy: function () {
    // Remove it.
    if (this.kind === 'learning-curve') {
      window.removeEventListener('resize', this.drawLearningCurve, false)
    } else if (this.kind === 'model-scatter') {
      window.removeEventListener('resize', this.drawModelScatter, false)
    }
  },
  methods: {
    drawLearningCurve: function () {
      if (!this.kind) return

      d3.select('#learning-curve-canvas').select('svg').remove() // Remove SVG if it has been created.
      const margin = this.margin
      const canvas = this.$refs.canvas
      this.canvas_width = canvas.clientWidth
      this.canvas_height = canvas.clientHeight
      const svg = d3.select('#learning-curve-canvas').append('svg').attr('id', 'learning-graph')
      let best_epoch_nth = 0
      let best_epoch_loss = 0
      let train_loss_list = []
      let valid_loss_list = []

      if (this.selectedModelObj) {
        const model = this.selectedModelObj
        const best_epoch = model.best_epoch_valid_result
        best_epoch_nth = (best_epoch && best_epoch.nth_epoch) ? best_epoch.nth_epoch : 0
        best_epoch_loss = (best_epoch && best_epoch.loss) ? best_epoch.loss : 0

        train_loss_list = model.train_loss_list
        valid_loss_list = model.valid_loss_list
        if (!train_loss_list) {
          train_loss_list = []
        }
        if (!valid_loss_list) {
          valid_loss_list = []
        }
      }

      const minX = this.endOfAxisXY.x.min
      const maxX = this.endOfAxisXY.x.max
      const minY = this.endOfAxisXY.y.min
      const maxY = this.endOfAxisXY.y.max

      // if line chart axis overflow, clip the graph
      svg
        .append('defs')
        .append('clipPath')
        .attr('id', 'learning-curve-clip')
        .append('rect')
        .attr('x', margin.left)
        .attr('y', margin.top)
        .attr('width', this.canvas_width - (margin.left + margin.right))
        .attr('height', this.canvas_height - (margin.top + margin.bottom))

      // Set size.
      svg.attr('width', this.canvas_width)
        .attr('height', this.canvas_height)
        .call(this.zoom)

      // Axis Settings
      this.scaleX = d3.scaleLinear().domain([minX, maxX])
        .range([0, this.canvas_width - margin.left - margin.right])
      this.scaleY = d3.scaleLinear().domain([minY, maxY])
        .range([this.canvas_height - margin.bottom - margin.top, 0])

      // Sublines.
      // Horizontal
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line axis')
        .call(
          d3.axisRight()
            .tickSize(this.canvas_width - margin.left - margin.right)
            .tickFormat('').ticks(5)
            .scale(this.scaleY)
        )
        .selectAll('.tick line')
        .style('stroke-dasharray', '2,2')

      // Vertical
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'grid-line axis')
        .call(
          d3.axisTop()
            .tickSize(margin.top + margin.bottom - this.canvas_height)
            .tickFormat('').ticks(5)
            .scale(this.scaleX)
        )
        .selectAll('.tick line')
        .style('stroke-dasharray', '2,2')

      this.axX = d3.axisBottom(this.scaleX).ticks(5)
      this.axY = d3.axisLeft(this.scaleY).ticks(5)
      this.gX = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, this.canvas_height - margin.bottom] + ')')
        .attr('class', 'axis')
        .call(this.axX)

      this.gY = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(this.axY)

      // Line graph
      const LineLayer = svg.append('g').attr('clip-path', 'url(#learning-curve-clip)')
      if (!this.switchTrainGraph) {
        this.TrainLine = LineLayer.append('path')
          .attr('id', 'train-line')
          .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
          .datum(train_loss_list)
          .attr('fill', 'none')
          .attr('stroke', train_color)
          .attr('stroke-width', 1.5)
          .attr('d', d3.line()
            .x((d, index) => { return this.scaleX(index + 1) })
            .y((d) => { return this.scaleY(d) })
            .curve(d3.curveLinear)
          )
      }
      if (!this.switchValidGraph) {
        this.ValidLine = LineLayer.append('path')
          .attr('id', 'valid-line')
          .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
          .datum(valid_loss_list)
          .attr('fill', 'none')
          .attr('stroke', valid_color)
          .attr('stroke-width', 1.5)
          .attr('d', d3.line()
            .x((d, index) => { return this.scaleX(index + 1) })
            .y((d) => { return this.scaleY(d) })
            .curve(d3.curveLinear)
          )
      }

      this.BestEpoc = LineLayer.append('line')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 1.4)
        .attr('opacity', 0.7)
        .attr('x1', this.scaleX(best_epoch_nth))
        .attr('y1', this.scaleY(maxY))
        .attr('x2', this.scaleX(best_epoch_nth))
        .attr('y2', this.scaleY(minY))

        .on('mouseenter', (d, index) => {
          // find svg id and get mouse position
          let x = d3.event.layerX
          let y = d3.event.layerY
          if (x >= this.canvas_width * 0.5) {
            x -= 105
          } else {
            x += 10
          }
          if (y >= this.canvas_height * 0.5) {
            // ↓約2行分
            y -= 60
          } else {
            y += 10
          }

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TooltipTextArray = [
            {
              'key': 'Best Epoch',
              'value': best_epoch_nth
            },
            {
              'key': 'Valid Loss',
              'value': best_epoch_loss.toFixed(2)
            }
          ]
          this.TooltipKind = 'valid'
          this.TooltipDisplay = true
        })
        .on('mouseleave', () => {
          this.TooltipDisplay = false
        })

      // Scatter graph
      const ScatterLayer = svg.append('g').attr('clip-path', 'url(#learning-curve-clip)')
      if (!this.switchTrainGraph) {
        this.TrainScatter = ScatterLayer.append('g')
          .attr('id', 'train-scatter')
          .selectAll('circle')
          .data(train_loss_list)
          .enter()
          .append('circle')
          .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
          .attr('cx', (d, index) => { return this.scaleX(index + 1) })
          .attr('cy', (d) => {
            return this.scaleY(d)
          })
          .attr('fill', train_color)
          .attr('r', 1.5)

          .on('mouseenter', (d, index) => {
            // find svg id and get mouse position
            let x = d3.event.layerX
            let y = d3.event.layerY
            if (x >= this.canvas_width * 0.5) {
              x -= 105
            } else {
              x += 10
            }
            if (y >= this.canvas_height * 0.5) {
              // ↓約2行分
              y -= 60
            } else {
              y += 10
            }

            this.TooltipLeft = x
            this.TooltipTop = y
            this.TooltipTextArray = [
              {
                'key': 'Epoch',
                'value': (index + 1)
              },
              {
                'key': 'Train Loss',
                'value': d.toFixed(2)
              }
            ]
            this.TooltipKind = 'train'
            this.TooltipDisplay = true
          })
          .on('mouseleave', () => {
            this.TooltipDisplay = false
          })
      }

      if (!this.switchValidGraph) {
        this.ValidScatter = ScatterLayer.append('g')
          .attr('id', 'valid-scatter')
          .selectAll('circle')
          .data(valid_loss_list)
          .enter()
          .append('circle')
          .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
          .attr('cx', (d, index) => { return this.scaleX(index + 1) })
          .attr('cy', (d) => { return this.scaleY(d) })
          .attr('fill', valid_color)
          .attr('r', 1.5)

          .on('mousemove', (d, index) => {
            let x = d3.event.layerX
            let y = d3.event.layerY
            if (x >= this.canvas_width * 0.5) {
              x -= 105
            } else {
              x += 10
            }
            if (y >= this.canvas_height * 0.5) {
              // ↓約2行分
              y -= 60
            } else {
              y += 10
            }

            this.TooltipLeft = x
            this.TooltipTop = y
            this.TooltipTextArray = [
              {
                'key': 'Epoch',
                'value': (index + 1)
              },
              {
                'key': 'Valid Loss',
                'value': d.toFixed(2)
              }
            ]
            this.TooltipKind = 'valid'
            this.TooltipDisplay = true
          })
          .on('mouseleave', () => {
            this.TooltipDisplay = false
          })
      }
    },

    drawModelScatter: function () {
      if (!this.kind) return
      if (!this.filteredAndGroupedModelListArray) return

      d3.select('#model-scatter-canvas').select('svg').remove() // Remove SVG if it has been created.
      const margin = this.margin
      const canvas = this.$refs.canvas
      this.canvas_width = canvas.clientWidth
      this.canvas_height = canvas.clientHeight
      const circle_radius = Math.min(this.canvas_width * 0.016, this.canvas_height * 0.016)
      const model_list = this.filteredAndGroupedModelListArray
      const svg = d3.select('#model-scatter-canvas').append('svg').attr('id', 'model-scatter-graph')

      // if line chart axis overflow, clip the graph
      svg
        .append('defs')
        .append('clipPath')
        .attr('id', 'model-scatter-clip')
        .append('rect')
        .attr('x', margin.left - 8)
        .attr('y', margin.top - 8)
        .attr('width', this.canvas_width - (margin.left + margin.right) + 16)
        .attr('height', this.canvas_height - (margin.top + margin.bottom) + 16)

      // Set size.
      svg
        .attr('width', this.canvas_width)
        .attr('height', this.canvas_height)
        .call(this.zoom)

      // Axis Settings
      // 最小値と最大値が同じ場合（モデルが1つ）、値を調整する処理を追加
      if (this.endOfAxisXY.x.min === this.endOfAxisXY.x.max) {
        this.endOfAxisXY.x.max = this.endOfAxisXY.x.max * 1.1
        this.endOfAxisXY.x.min = this.endOfAxisXY.x.min * 0.9
      }
      if (this.endOfAxisXY.y.min === this.endOfAxisXY.y.max) {
        this.endOfAxisXY.y.max = this.endOfAxisXY.y.max * 1.1
        this.endOfAxisXY.y.min = this.endOfAxisXY.y.min * 0.9
      }
      this.scaleX = d3.scaleLinear().domain([this.endOfAxisXY.x.min, this.endOfAxisXY.x.max])
        .range([0, this.canvas_width - margin.left - margin.right])

      this.scaleY = d3.scaleLinear().domain([this.endOfAxisXY.y.min, this.endOfAxisXY.y.max])
        .range([this.canvas_height - margin.bottom - margin.top, 0])

      // Sublines.
      // Horizontal
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(
          d3.axisRight()
            .tickSize(this.canvas_width - margin.left - margin.right)
            .tickFormat('')
            .scale(this.scaleY)
        )

      // Vertical
      svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(
          d3.axisTop()
            .tickSize(margin.top + margin.bottom - this.canvas_height)
            .tickFormat('')
            .scale(this.scaleX)
        )

      this.axX = d3.axisBottom(this.scaleX).ticks(5)
      this.axY = d3.axisLeft(this.scaleY).ticks(5)

      this.gX = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, this.canvas_height - margin.bottom] + ')')
        .attr('class', 'axis')
        .call(this.axX)
      this.gY = svg.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('class', 'axis')
        .call(this.axY)

      // Line graph
      const PlotLayer = svg.append('g').attr('clip-path', 'url(#model-scatter-clip)')
      let selected_model_id = 0
      if (this.selectedModelObj) {
        selected_model_id = this.selectedModelObj.id
      }

      // Check if it is in percent
      let Magnification = 1
      let Percent = ''
      if (this.percentMagnification) {
        Magnification = 100
        Percent = ' [%]'
      }

      // Plot Models.
      this.PlotModel = PlotLayer.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .selectAll('circle')
        .data(model_list)
        .enter()
        .append('circle')
        .attr('r', function (d) {
          if (d.id === selected_model_id) {
            return circle_radius * 1.6
          } else {
            return circle_radius
          }
        })
        .attr('cx', (m) => {
          const metric = m.getResultOfMetric1()
          if (metric.metric === 'RMSE') {
            metric.metric = 'RMSE (Root Mean Squared Error)'
          }
          this.axisNameXData = metric.metric + Percent
          if (metric.value === '-') metric.value = 0
          return this.scaleX(metric.value * Magnification)
        })
        .attr('cy', (m) => {
          const metric = m.getResultOfMetric2()
          if (metric.metric === 'MaxAbsErr') {
            metric.metric = 'Max Absolute Error'
          }
          this.axisNameYData = metric.metric + Percent
          if (metric.value === '-') metric.value = 0
          return this.scaleY(metric.value * Magnification)
        })
        .attr('fill', (m) => {
          return getAlgorithmColor(m.algorithm_id)
        })
        .on('mouseenter', (m, index) => {
          let x = d3.event.layerX
          let y = d3.event.layerY
          if (x >= this.canvas_width * 0.5) {
            x -= 105
          } else {
            x += 10
          }
          if (y >= this.canvas_height * 0.5) {
            // ↓約4行分
            y -= 95
          } else {
            y += 10
          }

          const metric1 = m.getResultOfMetric1()
          const metric2 = m.getResultOfMetric2()

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TooltipTextArray = [
            {
              'key': 'ID',
              'value': m.id
            },
            {
              'key': 'Alg',
              'value': this.algorithmTitleFunc(m.algorithm_id)
            },
            {
              'key': metric1.metric,
              'value': metric1.value
            },
            {
              'key': metric2.metric,
              'value': metric2.value
            },
          ]
          if (String(m.algorithm_id) === '4294967295') {
            this.TooltipKind = 'user-defined'
          } else {
            this.TooltipKind = String(m.algorithm_id).slice(-1)
          }
          // 学習が始まっていないModelは表示させない
          if (m.state !== undefined && (m.state === STATE.CREATED || m.state === STATE.CREATED)) {
            this.TooltipDisplay = false
          } else {
            this.TooltipDisplay = true
          }
        })
        .on('mouseleave', () => {
          this.TooltipDisplay = false
        })
        .on('click', (m) => {
          // 学習が始まっていないModelは表示させない
          if (m.state !== undefined && (m.state !== STATE.CREATED && m.state !== STATE.CREATED)) {
            this.$emit('update-sel-mod', m)
          }
        })
        // トレーニング中モデルの点滅処理
        .attr('class', (m) => {
          if (m.state !== undefined && m.state === STATE.STARTED) {
            return 'training-blink'
          } else if (m.state !== undefined && (m.state === STATE.CREATED || m.state === STATE.CREATED)) {
            // 学習が始まっていないModelは表示させない
            return 'opacity-0'
          } else {
            return ''
          }
        })
        .style('stroke', (m) => { return getAlgorithmColor(m.algorithm_id) })
    },

    zoomed: function () {
      if (this.kind === 'learning-curve') {
        const move_x = this.margin.left + d3.event.transform.x
        const move_y = this.margin.top + d3.event.transform.y
        this.TrainLine.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
        this.ValidLine.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
        this.TrainScatter.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
        this.ValidScatter.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
        this.BestEpoc.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
      } else {
        const move_x = d3.event.transform.x
        const move_y = d3.event.transform.y
        this.PlotModel.attr('transform', 'translate(' + [move_x, move_y] + ') scale(' + d3.event.transform.k + ')')
      }
      this.gX.call(this.axX.scale(d3.event.transform.rescaleX(this.scaleX)))
      this.gY.call(this.axY.scale(d3.event.transform.rescaleY(this.scaleY)))
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#learning-curve {
  width: 100%;
  height: 100%;
  position: relative;
  color: $gray;
  #title-epoch {
    position: absolute;
    top: calc(100% - #{$scatter-padding}*1.5);
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding});
    height: $scatter-padding;
    text-align: center;
    font-size: $fs-small;
  }
  #title-loss {
    position: absolute;
    top: 0;
    left: calc(#{$scatter-padding}*0.5);
    width: $scatter-padding;
    height: 100%;
    writing-mode: vertical-rl;
    text-align: center;
    font-size: $fs-small;
  }

  #learning-curve-canvas {
    position: absolute;
    top: $scatter-padding;
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding}*2);
    height: calc(100% - #{$scatter-padding}*2);
    .grid-line line {
      stroke: $light-gray;
    }
    .axis path {
      stroke: lightgray;
    }
    .axis line {
      stroke: $light-gray;
    }
    .tick {
      text {
        fill: $gray;
      }
    }
  }

  #model-scatter-canvas {
    position: absolute;
    top: $scatter-padding;
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding}*2);
    height: calc(100% - #{$scatter-padding}*2);
    .axis {
      path {
        stroke: lightgray;
      }
      line {
        stroke: $light-gray;
      }
    }
    .tick {
      text {
        fill: $gray;
      }
      line {
        stroke-dasharray: 2, 2;
      }
    }
    .opacity-0 {
      opacity: 0;
    }
  }
}
</style>
