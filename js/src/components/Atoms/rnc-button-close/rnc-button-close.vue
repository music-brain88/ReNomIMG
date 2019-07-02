<template>
  <div
    v-if="display"
    id="model-buttons"
  >
    <i
      class="fa fa-times"
      v-on="$listeners"
    />
  </div>
</template>

<script>
export default {
  name: 'RncButtonClose',
  props: {
    display: { type: Boolean, default: true },
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

#model-buttons {
  i {
    color: lightgray;
  }
  .fa-times {
    &:hover {
      color: $model-item-remove-button-color;
      cursor: pointer;
    }
  }
  .fa:active {
    color: gray;
  }
  .fa{display:inline-block;font:normal normal normal 14px/1 FontAwesome;font-size:inherit;text-rendering:auto;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
}

</style>
