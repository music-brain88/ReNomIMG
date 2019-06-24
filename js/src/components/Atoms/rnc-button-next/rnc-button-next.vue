<template>
  <div class="pager-arrow">
    <i
      :class="['fa-caret-' + kindClass]"
      class="fa fa-caret-left"
      aria-hidden="true"
      v-on="$listeners"
    />
  </div>
</template>

<script>
export default {
  name: 'RncButtonNext',
  props: {
    kind: {
      type: String,
      default: 'right'
    },
  },
  data () {
    return {
      LEFT: 'left',
      RIGHT: 'right'
    }
  },
  computed: {
    kindClass () {
      const kindClass = this.kind.toLowerCase()
      if (this.LEFT !== kindClass && this.RIGHT !== kindClass) {
        throw new Error(`The 'kind' property must be: '${this.LEFT}' or '${this.RIGHT}' but is: ${kindClass}`)
      }
      return kindClass
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

.pager-arrow {
  font-size: 150%;
  align-items: center;
  cursor: pointer;
  color: lightgray;
  transition: all 0.02s;
  &:hover {
    color: gray;
  }
}
</style>
