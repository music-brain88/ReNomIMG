<template>
  <transition
    v-if="display"
    name="fade"
  >
    <div
      :style="'top:' + top + 'px; left:' + left + 'px;'"
      class="tooltip-position"
    >
      <div
        :class="'color-' + kind.toLowerCase()"
        :style="'opacity:' + Opacity + ';'"
        class="scatter-canvas"
      >
        <div
          v-for="(item, index) in textArray"
          :key="index"
        >
          <div>
            {{ item.key }} : {{ item.value }}
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>

export default {
  name: 'RncTooltip',
  props: {
    kind: {
      type: [Number, String],
      default: 'no-model',
      validator: val => ['no-model', 'train', 'valid', 'user-defined', '0', '1', '2', '3', '4', '5'].includes(val)
    },
    top: {
      type: Number,
      default: 0
    },
    left: {
      type: Number,
      default: 0
    },
    display: {
      type: Boolean,
      default: false
    },
    textArray: {
      type: Array,
      default: undefined
    }
  },
  data: function () {
    return {
      Opacity: 0
    }
  },
  watch: {
    display: function () {
      this.displayOpacity()
    }
  },

  mounted: function () {
    this.displayOpacity()
  },
  methods: {
    displayOpacity: function () {
      if (this.display === true) {
        this.Opacity = 0.9
      } else {
        this.Opacity = 0
      }
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

.tooltip-position{
  position: absolute;
  z-index: 99;
  .scatter-canvas {
    -webkit-transition: all .2s;
    -moz-transition: all .2s;
    -ms-transition: all .2s;
    -o-transition: all .2s;
    transition: all .2s;
    min-width: 80px;
    color: white;
    padding: 10px;
    font-size: 0.8rem;
    line-height: 1.1rem;
    text-align: left;
  }
}
</style>
