<template>
  <transition name="modal">
    <div id="modal" v-if="show">
      <div id="modal-mask" @click="toggle">
      </div>
      <div id="modal-content">
        <add-both v-if="show_modal.add_both"/>
        <add-dataset v-if="show_modal.add_dataset"/>
        <add-filter v-if="show_modal.add_filter"/>
      </div>
    </div>
  </transition>
</template>

<script>
import { mapGetters, mapMutations, mapState } from 'vuex'
import ModalAddBoth from '@/components/page/train_page/modal_add_both.vue'
import ModalAddDataset from '@/components/page/train_page/modal_add_dataset.vue'
import ModalAddFilter from '@/components/page/train_page/modal_add_filter.vue'

export default {
  name: 'Modal',
  components: {
    'add-both': ModalAddBoth,
    'add-dataset': ModalAddDataset,
    'add-filter': ModalAddFilter
  },
  computed: {
    ...mapState(['show_modal']),
    show: function () {
      return Object.values(this.show_modal).some((d) => d)
    }
  },
  methods: {
    ...mapMutations(['showModal']),
    toggle: function () {
      this.showModal({'all': false})
    }
  }
}
</script>

<style lang='scss'>
#modal {
  position: fixed;
  top: 0;
  left: 0;
  #modal-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: $modal-mask-color;
  }
  #modal-content {
    position: fixed;
    left: calc((100vw - 60vw) / 2);
    top: calc((100vh - 60vh) / 2);
    width: 60vw;
    height: 60vh;
    background-color: white;
    padding: $modal-content-padding;
  }
}
</style>
