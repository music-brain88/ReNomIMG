<template>
  <rnc-modal-fixmask
    @modal-ok="onClick"
    @modal-cancel="hide"
  >
    <template slot="modal-contents">
      <span v-html="modal_msg"/>
    </template>
  </rnc-modal-fixmask>
</template>

<script>
import { mapMutations, mapState } from 'vuex'
import RncModalFixmask from './../../Atoms/rnc-modal-fixmask/rnc-modal-fixmask.vue'

export default {
  name: 'RncModalConfirm',
  components: {
    'rnc-modal-fixmask': RncModalFixmask
  },
  computed: {
    ...mapState([
      'modal_msg',
      'confirm_modal_callback_function'
    ])
  },
  methods: {
    ...mapMutations(['hideConfirm']),
    hide: function () {
      this.hideConfirm()
    },
    onClick: function () {
      if (this.confirm_modal_callback_function) {
        this.confirm_modal_callback_function()
      }
      this.hide()
    }
  }
}
</script>

<style lang='scss'>
</style>
