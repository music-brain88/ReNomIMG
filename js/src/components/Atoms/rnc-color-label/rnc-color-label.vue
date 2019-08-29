<template>
  <div
    :class="[isToggleLabel ? 'cursor-pointer':'', (isToggleLabel && lineThrough)?'line-through':'']"
    class="rnc-color-label"
    v-on="$listeners"
  >
    <span
      :class="colorClass"
      class="color-box"
    />
    {{ title }}
  </div>
</template>

<script>
export default {
  name: 'RncColorLabel',
  props: {
    title: {
      type: String,
      default: 'Undefined'
    },
    colorClass: {
      type: String,
      default: 'color-0',
      validator: val => ['color-train', 'color-valid', 'color-created', 'color-reserved', 'color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5'].includes(val)
    },
    lineThrough: {
      type: Boolean,
      default: false
    },
    isToggleLabel: {
      type: Boolean,
      default: false
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

.rnc-color-label{
  font-family: $component-font-family;
  font-size: $fs-small;
  color: $gray;
  height: 100%;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  .color-box {
    width: 10px;
    height: 10px;
    margin-right: 3px;
  }
}
.line-through {
  text-decoration: line-through;
}
.cursor-pointer {
  cursor: pointer;
}
</style>
