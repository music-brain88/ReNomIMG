<template>
  <div id="learning-curve">
    <div id="title-epoch">
      {{ AxisNameXData }}
    </div>
    <div id="title-loss">
      {{ AxisNameYData }}
    </div>
    <div
      ref="canvas"
      :id="idKind"
    >
      <rnc-tooltip
        :top="TooltipTop"
        :left="TooltipLeft"
        :display="TooltipDisplay"
        :text-array="TextArray"
        :kind="TooltipKind"
      />
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import * as d3 from 'd3'
import { train_color, valid_color, algorithm_colors } from './../../../const'
import RncTooltip from './../../Atoms/rnc-tooltip/rnc-tooltip.vue'

export default {
  name: 'RncGridXY',
  components: {
    'rnc-tooltip': RncTooltip
  },
  props: {
    kind: {
      type: String,
      default: 'learning-curve',
      validator: val => ['learning-curve', 'model-scatter'].includes(val),
    },
    AxisNameX: {
      type: String,
      default: '',
    },
    AxisNameY: {
      type: String,
      default: '',
    },
    SelectedModelObj: {
      type: Object,
      default: function () { return undefined },
    },
    FilteredAndGroupedModelListArray: {
      type: Array,
      default: function () { return undefined },
    },
    switchTrainGraph: {
      type: Boolean,
      default: false,
    },
    switchValidGraph: {
      type: Boolean,
      default: false,
    }
  },
  data: function () {
    return {
      tooltip: null,
      train_graph_flg: true,
      valid_graph_flg: true,
      TooltipTop: 0,
      TooltipLeft: 0,
      TooltipDisplay: false,
      TextArray: undefined,
      TooltipKind: 'no-model',
      AxisNameXData: '',
      AxisNameYData: '',

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
      const model = this.SelectedModelObj
      if (undefined !== model) {
        // TODO muraishi: train_loss_list
        return model.train_loss_list
      } else {
        return []
      }
    },
    idKind () {
      return `${this.kind}-canvas`
    },
    ...mapGetters([
      'getAlgorithmTitleFromId',
      'getSelectedModel'
    ]),
    zoom: function () {
      const zoom = d3.zoom()
        .scaleExtent([1, 2])
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
    FilteredAndGroupedModelListArray: function () {
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
    getSelectedModel: function () {
      const svg = d3.select('#model-scatter-canvas').select('svg')
      if (!svg) return
      const canvas = document.getElementById('model-scatter-canvas')
      if (!canvas) return
      const canvas_width = canvas.clientWidth
      const canvas_height = canvas.clientHeight
      const circle_radius = Math.min(canvas_width * 0.02, canvas_height * 0.02)
      svg.selectAll('circle')
        .attr('r', (m) => {
          const model = this.getSelectedModel
          if (model === m) {
            return circle_radius * 1.6
          } else {
            return circle_radius
          }
        })
    }
  },
  mounted: function () {
    if (this.kind === 'learning-curve') {
      this.AxisNameXData = this.AxisNameX
      this.AxisNameYData = this.AxisNameY
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
    ...mapMutations(['setSelectedModel']),
    // ADD muraishi
    ...mapActions([
      'loadModelsOfCurrentTaskDetail',
      'loadDatasetsOfCurrentTaskDetail']),

    drawLearningCurve: function () {
      if (!this.kind) return
      if (!this.SelectedModelObj) return

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

      if (this.SelectedModelObj) {
        const model = this.SelectedModelObj
        // TODO muraishi : best_epoch_valid_result
        const best_epoch = model.best_epoch_valid_result
        best_epoch_nth = (best_epoch && best_epoch.nth_epoch) ? best_epoch.nth_epoch : 0
        best_epoch_loss = (best_epoch && best_epoch.loss) ? best_epoch.loss : 0

        // TODO muraishi : train_loss_list
        // TODO muraishi : valid_loss_list
        train_loss_list = model.train_loss_list
        valid_loss_list = model.valid_loss_list
        if (!train_loss_list) {
          train_loss_list = []
        }
        if (!valid_loss_list) {
          valid_loss_list = []
        }
      }
      const learning_epoch = train_loss_list.length
      let maxX = Math.max(learning_epoch + 1, 10)
      maxX = Math.ceil(maxX / 5) * 5
      const minX = 0
      let maxY = Math.max((Math.max.apply(null, [...train_loss_list, ...valid_loss_list]) * 1.1), 1)
      maxY = Math.ceil(maxY)
      let minY = Math.min(Math.min.apply(null, [...train_loss_list, ...valid_loss_list]), 0)
      minY = Math.floor(minY)

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

      if (!this.tooltip) {
        // Ensure only one tooltip is exists.
        this.tooltip = d3.select('#learning-curve-canvas')
          .append('div')
          .append('display', 'none')
          .style('position', 'absolute')
      }

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

      this.BestEpoc = LineLayer.append('line')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 1)
        .attr('opacity', 0.7)
        .attr('x1', this.scaleX(best_epoch_nth + 1))
        .attr('y1', this.scaleY(maxY))
        .attr('x2', this.scaleX(best_epoch_nth + 1))
        .attr('y2', this.scaleY(minY))

        .on('mouseenter', (d, index) => {
          // find svg id and get mouse position
          const x = d3.event.layerX + 10
          const y = d3.event.layerY + 10

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TextArray = [
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
          const x = d3.event.layerX + 10
          const y = d3.event.layerY + 10

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TextArray = [
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
          const x = d3.event.layerX + 10
          const y = d3.event.layerY + 10

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TextArray = [
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

      // d3.select('#learning-curve-canvas')
      //   .on('contextmenu', this.resetZoom)
    },

    drawModelScatter: function () {
      if (!this.kind) return
      if (!this.FilteredAndGroupedModelListArray) return

      d3.select('#model-scatter-canvas').select('svg').remove() // Remove SVG if it has been created.
      const margin = this.margin
      const canvas = this.$refs.canvas
      this.canvas_width = canvas.clientWidth
      this.canvas_height = canvas.clientHeight
      const circle_radius = Math.min(this.canvas_width * 0.02, this.canvas_height * 0.02)
      const model_list = this.FilteredAndGroupedModelListArray
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
      this.scaleX = d3.scaleLinear().domain([0, 100])
        .range([0, this.canvas_width - margin.left - margin.right])

      this.scaleY = d3.scaleLinear().domain([0, 100])
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

      if (!this.tooltip) {
        this.tooltip = d3.select('#model-scatter-canvas')
          .append('div')
          .style('display', 'none')
          .style('position', 'absolute')
      }

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

      // Plot Models.
      this.PlotModel = PlotLayer.append('g')
        .attr('transform', 'translate(' + [margin.left, margin.top] + ')')
        .selectAll('circle')
        .data(model_list)
        .enter()
        .append('circle')
        .attr('r', circle_radius)
        .attr('cx', (m) => {
          // TODO: Modify data distribution
          const metric = m.getResultOfMetric1()
          this.AxisNameXData = metric.metric + ' [%]'
          if (metric.value === '-') metric.value = 0
          const total_width = this.canvas_width - margin.left - margin.right
          const rescaled_point_x = metric.value * total_width
          return rescaled_point_x
        })
        .attr('cy', (m) => {
          // TODO: Modify data distribution
          const metric = m.getResultOfMetric2()
          this.AxisNameYData = metric.metric + ' [%]'
          if (metric.value === '-') metric.value = 0
          const total_height = this.canvas_height - margin.top - margin.bottom
          const rescaled_point_y = (1 - metric.value) * total_height
          return rescaled_point_y
        })
        .attr('fill', (m) => {
          if (String(m.algorithm_id).slice(-1) === '0') {
            return algorithm_colors.color_0
          } else if (String(m.algorithm_id).slice(-1) === '1') {
            return algorithm_colors.color_1
          } else if (String(m.algorithm_id).slice(-1) === '2') {
            return algorithm_colors.color_2
          } else if (String(m.algorithm_id).slice(-1) === '3') {
            return algorithm_colors.color_3
          } else if (String(m.algorithm_id).slice(-1) === '4') {
            return algorithm_colors.color_4
          } else if (String(m.algorithm_id).slice(-1) === '5') {
            return algorithm_colors.color_5
          } else {
            return algorithm_colors.color_no_model
          }
        })
        .on('mouseenter', (m, index) => {
          let x = d3.event.layerX + 10
          let y = d3.event.layerY + 10
          if (x >= this.canvas_width * 0.8) {
            x -= 100
          }
          if (y >= this.canvas_height * 0.8) {
            y -= 80
          }

          const metric1 = m.getResultOfMetric1()
          const metric2 = m.getResultOfMetric2()

          this.TooltipLeft = x
          this.TooltipTop = y
          this.TextArray = [
            {
              'key': 'ID',
              'value': m.id
            },
            {
              'key': 'Model',
              'value': this.getAlgorithmTitleFromId(m.algorithm_id)
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
          this.TooltipKind = String(m.algorithm_id).slice(-1)
          this.TooltipDisplay = true
        })
        .on('mouseleave', () => {
          this.TooltipDisplay = false
        })
        .on('click', (m) => {
          // CHANGE muraishi
          this.clickedModelItem(m)
        })
        // .on('click', (m) => {
        //   this.setSelectedModel(m)
        // })
      // d3.select('#model-scatter-canvas')
      //   .on('contextmenu', this.resetZoom)
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
    },
    // TODO:おそらく使われていないためコメントアウト
    // resetZoom: function () {
    //   svg.transition().duration(1000).call(zoom.transform, d3.zoomIdentity)
    //   if (this.kind == "learning-curve") {
    //     this.TrainLine.attr('transform', 'translate(' + [this.margin.left, this.margin.top] + ')')
    //     this.ValidLine.attr('transform', 'translate(' + [this.margin.left, this.margin.top] + ')')
    //     this.TrainScatter.attr('transform', 'translate(' + [this.margin.left, this.margin.top] + ')')
    //     this.ValidScatter.attr('transform', 'translate(' + [this.margin.left, this.margin.top] + ')')
    //   } else {
    //     this.PlotModel.attr('transform', 'translate(' + [this.margin.left, this.margin.top] + ')')
    //   }
    //   d3.event.preventDefault()
    // }
    clickedModelItem: function (model) {
      this.loadModelsOfCurrentTaskDetail(model.id)
      this.loadDatasetsOfCurrentTaskDetail(model.dataset_id)
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
  color: $component-font-color-title;
  #title-epoch {
    position: absolute;
    top: calc(100% - #{$scatter-padding}*1.5);
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding});
    height: $scatter-padding;
    text-align: center;
    font-size: $component-font-size-small;
  }
  #title-loss {
    position: absolute;
    top: 0;
    left: calc(#{$scatter-padding}*0.5);
    width: $scatter-padding;
    height: 100%;
    writing-mode: vertical-rl;
    text-align: center;
    font-size: $component-font-size-small;
  }

  #learning-curve-canvas {
    position: absolute;
    top: $scatter-padding;
    left: $scatter-padding;
    width: calc(100% - #{$scatter-padding}*2);
    height: calc(100% - #{$scatter-padding}*2);
    .grid-line line {
      stroke: $scatter-grid-color;
    }
    .axis path {
      stroke: lightgray;
    }
    .axis line {
      stroke: $scatter-grid-color;
    }
    .tick {
      text {
        fill: $component-font-color-title;
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
        stroke: $scatter-grid-color;
      }
    }
    .tick {
      text {
        fill: $component-font-color-title;
      }
      line {
        stroke-dasharray: 2, 2;
      }
    }
  }
}
</style>
