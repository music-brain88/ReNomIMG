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
      :class="getColor(model[0])"
      :style="getWidth(model[1])"
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
const RESERVED = -1
const CREATED = -2

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
    // think if better to use getter getAlgorithmColor()
    getColor: function (model_idx) {
      model_idx = Number(model_idx)
      if (model_idx === CREATED) {
        return 'color-created'
      } else if (model_idx === RESERVED) {
        return 'color-reserved'
      } else {
        return `color-${model_idx % 10}`
      }
    },
    getWidth: function (model_rato) {
      return `width : ${model_rato * 100}%;`
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
      min-width: 10%;
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
