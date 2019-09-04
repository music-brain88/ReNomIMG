<template>
  <div class="rnc-bar-progress">
    <div id="bar-background">
      <div
        id="bar-front"
        :style="getWidthOfBar"
        :class="[colorClass, barClass]"
      />
    </div>
  </div>
</template>

<script>
export default {
  name: 'RncBarProgress',
  props: {
    barClass: {
      type: String,
      default: 'validating',
      validator: val => ['validating', 'training'].includes(val)
    },
    colorClass: {
      type: String,
      default: 'color-0',
      // validator: val => ['color-created', 'color-reserved', 'color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5'].includes(val)
    },
    totalBatch: {
      type: Number,
      default: 0
    },
    currentBatch: {
      type: Number,
      default: 0
    }
  },
  data: function () {
    return {
      onHovering: false,
    }
  },
  computed: {
    getWidthOfBar: function () {
      if (this.barClass === 'validating') {
        return {
          width: '20%'
        }
      } else {
        if (this.totalBatch === 0) {
          return {
            width: 0 + '%'
          }
        } else {
          return {
            width: (this.currentBatch / this.totalBatch) * 100 + '%'
          }
        }
      }
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

.rnc-bar-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  #bar-background {
    width: 100%;
    height: calc(100% - #{$progress-bar-margin}*2);
    background-color: $light-gray;
    #bar-front.training {
      position: relative;
      top: 0;
      left: 0;
      height: 100%;
      transition: width 300ms;
    }
    #bar-front.validating {
      position: relative;
      top: 0;
      left: 0;
      height: 100%;
      // transition: width 300ms;

      animation: move-bar 1.5s;
      animation-iteration-count: infinite;
      animation-timing-function: linear;
      animation-fill-mode: both;
      animation-delay: 0.1s;
    }

    @keyframes move-bar {
      0% {
        transform: translateX(-50%) scaleX(0);
      }
      20% {
        transform: translateX(0%) scaleX(1);
      }
      80% {
        transform: translateX(400%) scaleX(1);
      }
      100% {
        transform: translateX(450%) scaleX(0);
      }
    }
  }
}
</style>
