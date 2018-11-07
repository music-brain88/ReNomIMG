<template>
  <div id="modal-add-both">
    <div id="modal-tab">
      <div id="add-model-tab" class="modal-tab" @click="showAddModel">
        Model
      </div>
      <div id="add-dataset-tab" class="modal-tab" @click="showAddDataset">
        Dataset
      </div>
    </div>
    <div id="add-component">
      <add-model v-if="isAddModelShown"/>
      <add-dataset v-else/>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState } from 'vuex'
import ModalAddModel from '@/components/page/train_page/modal_add_model.vue'
import ModalAddDataset from '@/components/page/train_page/modal_add_dataset.vue'

export default {
  name: 'ModalAddBoth',
  components: {
    'add-model': ModalAddModel,
    'add-dataset': ModalAddDataset
  },
  data: function () {
    return {
      isAddModelShown: true
    }
  },
  computed: {
    ...mapState(['show_modal'])
  },
  created: function () {
  },
  methods: {
    ...mapMutations(['showModal']),
    showAddModel: function () {
      this.isAddModelShown = true
    },
    showAddDataset: function () {
      this.isAddModelShown = false
    },
  }
}
</script>

<style lang='scss'>
#modal-add-both {
  width: 100%;
  height: 100%;
  #modal-tab {
    height: 10%;
    width: 100%;
    .modal-tab {
      display: inline-block;
      height: 100%;
      width: 30%;
    }
  }
  #add-component {
    height: 90%;
    width: 100%;
  }
}
</style>
