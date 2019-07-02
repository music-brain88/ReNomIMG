<template>
  <div
    v-if="display"
    id="button-stop-area"
  >
    <i
      class="fa fa-stop-circle-o"
      v-on="$listeners"
    />
  </div>
</template>

<script>
export default {
  name: 'RncButtonStop',
  props: {
    display: { type: Boolean, default: true },
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

#button-stop-area {
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: $progress-bar-stop-button-color;
  font-size: $progress-bar-stop-button-font-size;
  i {
    cursor: pointer;
    &:hover {
      color: $progress-bar-stop-button-color-hover;
    }
  }
}
</style>
