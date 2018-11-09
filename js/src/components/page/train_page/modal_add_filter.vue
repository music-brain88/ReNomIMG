<template>
  <div id="modal-add-filter">
    This is filter
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'

export default {
  name: 'ModalAddFilter',
  components: {
  },
  computed: {
    ...mapState([]),
    ...mapGetters([
      'getCurrentTask',
      'getAlgorithmList',
      'getAlgorithmParamList',
      'getAlgorithmIdFromTitle'
    ]),
  },
  data: function () {
    return {
    }
  },
  created: function () {

  },
  methods: {
  }
}
</script>

<style lang='scss'>
#modal-add-filter {
  width: 100%;
  height: 100%;
}
</style>
