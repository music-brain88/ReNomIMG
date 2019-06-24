<template>
  <rnc-modal-fixmask @modal-ok="hide">
    <template slot="modal-contents">
      {{ error_msg }}
    </template>
    <template slot="cancelbutton">
      <div class="none" />
    </template>
  </rnc-modal-fixmask>
</template>

<script>
import { mapMutations, mapState } from 'vuex'
import RncModalFixmask from './../../Atoms/rnc-modal-fixmask/rnc-modal-fixmask.vue'

export default {
  name: 'RncModalAlert',
  components: {
    'rnc-modal-fixmask': RncModalFixmask
  },
  computed: {
    ...mapState(['error_msg'])
  },
  methods: {
    ...mapMutations(['hideAlert']),
    hide: function () {
      this.hideAlert()
    }
  }
}
</script>

<style lang='scss'>
.none {
  display: none;
}
</style>
