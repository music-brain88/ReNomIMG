<template>
  <div
    id="component-frame"
    :style="styles">
    <div id="component-header">
      <slot name="header-slot"/>
    </div>
    <div id="frame-content">
      <slot/>
    </div>
  </div>
</template>

<script>

export default {
  name: 'ComponentFrame',
  props: {
    widthWeight: { // (Max 5, Min 1)
      type: Number,
      default: 1,
    },
    heightWeight: { // (Max 5, Min 1)
      type: Number,
      default: 1,
    }
  },
  computed: {
    styles () {
      return {
        '--width-weight': this.widthWeight,
        '--height-weight': this.heightWeight
      }
    }
  },
}
</script>

<style lang='scss'>

#component-frame {
  --width-weight: 1;
  --height-weight: 1;

  width: calc(var(--width-weight) * #{$component-block-width}
    - #{$component-block-margin}*2);
  height: calc(var(--height-weight) * #{$component-block-height}
    - #{$component-block-margin}*2
    + #{$component-header-margin-bottom}
    + #{$component-header-height});
  min-height: calc(var(--height-weight) * #{$component-block-height-min}
    - #{$component-block-margin}*2
    + #{$component-header-margin-bottom}
    + #{$component-header-height});

  margin: $component-block-margin;

  #component-header {
    width: 100%;
    height: $component-header-height;
    min-height: $component-header-min-height;
    margin-bottom: $component-header-margin-bottom;
    padding-left: $component-block-padding-left;
    background-color: $component-header-background-color;
    color: $component-header-font-color;
    line-height: normal;
    font-family: $component-header-font-family;
    font-size: $component-header-font-size;
    div {
      font-family: $component-header-font-family;
    }
    display: flex;
    align-items: center; /* 縦方向中央揃え */
    justify-content: space-between;
  }

  #frame-content{
    background-color: $component-background-color;
    height: calc(100% - #{$component-header-margin-bottom} - #{$component-header-height});
    width: 100%;
  }
}
</style>
