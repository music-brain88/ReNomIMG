<template>
  <div
    :style="styles"
    class="rnc-title-frame"
  >
    <div class="component-header">
      <slot name="header-slot" />
    </div>
    <div class="frame-content">
      <slot name="content-slot" />
    </div>
  </div>
</template>

<script>

export default {
  name: 'RncTitleFrame',
  props: {
    widthWeight: { // (Max 5, Min 1)
      type: Number,
      default: 1,
    },
    heightWeight: { // (Max 5, Min 1)
      type: Number,
      default: 0,
    }
  },
  computed: {
    styles () {
      let height = ''
      if (!this.heightWeight) {
        height = '100%'
      }
      return {
        '--width-weight': this.widthWeight,
        '--height-weight': this.heightWeight,
        'height': height
      }
    }
  },
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.rnc-title-frame {
  --width-weight: 1;
  --height-weight: 0;
  padding: $component-block-padding;
  width: calc(
    var(--width-weight) * #{$component-block-width}
  );
  height: calc(
    var(--height-weight) * #{$component-block-height}
    + #{$component-block-padding} * 2
    + #{$component-header-margin-bottom}
    + #{$component-header-height}
  );
  min-width: calc(
    1 * #{$component-block-width}
    - #{$component-block-padding} * 2
  );
  min-height: calc(
    var(--height-weight) * #{$component-block-height-min}
    + #{$component-block-padding} * 2
    + #{$component-header-margin-bottom}
    + #{$component-header-height}
  );
  .component-header {
    width: 100%;
    height: $component-header-height;
    min-height: $component-header-min-height;
    margin-bottom: $component-header-margin-bottom;
    padding-left: $component-block-padding-left;
    background-color: $dark-blue;
    color: $white;
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
  .frame-content {
    background-color: $white;
    height: calc(
      100% - #{$component-header-margin-bottom}
      - #{$component-header-height}
    );
    width: 100%;
  }
}
</style>
