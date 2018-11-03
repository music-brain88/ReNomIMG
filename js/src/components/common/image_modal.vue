<template>
  <div id="image-modal">
  </div>
</template>

<script>
import {mapGetters} from 'vuex'

export default {
  name: 'ImageModal',
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
#image-modal {
  width: 100%;
  height: 100%;
  background-color: $modal-mask-color;
}
</style>
