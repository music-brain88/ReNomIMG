<template>
  <div id="modal-add-both">
    <div id="modal-tab">
      <div id="add-model-tab" class="modal-tab"
        v-bind:class="{selected: isAddModelShown}" @click="showAddModel">
        Model
      </div>
      <div id="add-dataset-tab" class="modal-tab"
        v-bind:class="{selected: !isAddModelShown}" @click="showAddDataset">
        Dataset
      </div>
      <div id="space-tab">
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
      isAddModelShown: true,
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
    display: flex;
    height: 8%;
    width: 100%;
    .modal-tab {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      width: 25%;
      background-color: $modal-tab-color-not-selected;
      color: white;
    }
    .selected {
      color: black;
      background-color: white;
      border-top: solid 1px lightgray;
      border-left: solid 1px lightgray;
      border-right: solid 1px lightgray;
    }
    #space-tab {
      height: 100%;
      width: 50%;
      border-bottom: solid 1px lightgray;
    }
  }

  #add-component {
    height: 90%;
    width: 100%;
    padding: 10px;
  }
}
</style>
