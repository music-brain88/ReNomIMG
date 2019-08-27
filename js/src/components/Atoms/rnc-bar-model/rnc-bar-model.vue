<template>
  <div
    v-if="modelInfo && modelInfo.length != 0"
    id="rnc-bar-model"
    @mouseenter="onHovering=true"
    @mouseleave="onHovering=false"
  >
    <section
      v-for="(model, key) in modelInfo"
      :key="key"
      :style="getStyle(model)"
    >
      <transition name="fade">
        <div
          v-if="onHovering"
          class="bar-content"
        >
          {{ model[2] }}
        </div>
      </transition>
    </section>
  </div>

  <div
    v-else
    id="rnc-bar-model"
  >
    <section
      :class="'color-no-model'"
      style="width: 100%;"
    >
      <span class="bar-content">
        No Model
      </span>
    </section>
  </div>
</template>

<script>
import { getAlgorithmColor } from './../../../utils.js'

export default {
  name: 'RncBarModel',
  props: {
    modelInfo: {
      type: Array,
      default: function () { return undefined }
    }
  },
  data: function () {
    return {
      onHovering: false,
    }
  },
  methods: {
    getStyle: function (model) {
      const model_idx = model[0]
      const model_rato = model[1]
      return `width: ${model_rato * 100}%; background-color: ${getAlgorithmColor(model_idx)};`
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

  #rnc-bar-model {
    width: 100%;
    height: 100%;
    display: flex;
    overflow: hidden;
    section {
      height: 100%;
      min-width: 4%;
      line-height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      color: white;
      .bar-content {
        font-size: $fs-small;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
      }
    }
    section:last-of-type {
      flex: auto;
    }
}
</style>
