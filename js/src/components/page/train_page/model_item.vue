<template>
  <div id="model-item">
  </div>
</template>

<script>
import {mapGetters} from 'vuex'

export default {
  name: 'ModelItem',
  components: {

  },
  props: {

  },
  computed: {

  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#model-item {
  width: calc(100% - #{$model-item-margin}*2);
  height: $model-item-height;
  min-height: $model-item-height-min;
  margin: $model-item-margin;
  background-color: red;
}

</style>
