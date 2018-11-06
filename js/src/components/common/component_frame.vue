<template>
  <div id="component-frame" :style="styles">
    <div id="component-header">
      <slot name="header-slot">
      </slot>
    </div>
    <div id="frame-content">
      <slot>
      </slot>
    </div>
  </div>
</template>

<script>

export default {
  name: 'ComponentFrame',
  props: {
    widthWeight: Number, // (Max 5, Min 1)
    heightWeight: Number // (Max 5, Min 1)
  },
  computed: {
    styles () {
      return {
        '--width-weight': this.widthWeight,
        '--height-weight': this.heightWeight
      }
    }
  },
  created: function () {

  },
  methods: {

  }
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
    height: $component-header-height;
    min-height: $component-header-min-height;
    margin-bottom: $component-header-margin-bottom;
    background-color: $component-header-background-color;
    color: $component-header-font-color;
    font-family: $component-header-font-family;
  }

  #frame-content{
    background-color: $component-background-color; 
    height: calc(100% - #{$component-header-margin-bottom} - #{$component-header-height});
    width: 100%;
  }
}
</style>
