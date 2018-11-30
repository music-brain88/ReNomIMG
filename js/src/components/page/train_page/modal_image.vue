<template>
  <div id="image-modal">
    <img :src="modal_image"/>
      {{modal_prediction}}
  </div>
</template>

<script>
import { mapGetters, mapState } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ModalImage',
  components: {
  },
  computed: {
    ...mapState([
      'datasets',
      'modal_image',
      'modal_prediction',
      'modal_target'
    ]),
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
}
</style>
