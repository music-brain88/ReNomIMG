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
    heightWeight: Number, // (Max 5, Min 1)
  },
  computed: {
    styles () {
      return {
        '--width-weight': this.widthWeight,
        '--height-weight': this.heightWeight
      }
    }
  },
  created: {

  },
  methods: {

  },
}
</script>

<style lang='scss'>

#component-frame {
  --width-weight: 1;
  --height-weight: 1;
  width: calc(var(--width-weight) * #{$component-block-width});
  height: calc(var(--height-weight) * #{$component-block-height} + #{$component-header-height});
  margin: $component-block-margin;

  #component-header { 
    height: $component-header-height;
    margin-bottom: $component-header-margin-bottom;
    background-color: $component-header-background-color;
  }

  #frame-content{
    background-color: $component-background-color; 
    height: calc(100% - #{$component-header-margin-bottom} - #{$component-header-height});
  }
}
</style>
