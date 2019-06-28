<template>
  <div id="modal-add-both">
    <div id="modal-tab">
      <div
        id="add-model-tab"
        :class="{selected: isAddModelShown}"
        class="modal-tab"
        @click="showAddModel"
      >
        Model
      </div>
      <div
        id="add-dataset-tab"
        :class="{selected: !isAddModelShown}"
        class="modal-tab"
        @click="showAddDataset"
      >
        Dataset
      </div>
      <div id="space-tab" />
    </div>
    <div id="add-component">
      <rnc-modal-model v-if="isAddModelShown" />
      <rnc-modal-dataset v-else />
    </div>
  </div>
</template>

<script>
import { mapMutations } from 'vuex'
import RncModalTogglemask from '../../Atoms/rnc-modal-togglemask/rnc-modal-togglemask.vue'
import RncModalModel from '../rnc-modal-model/rnc-modal-model.vue'
import RncModalDataset from '../rnc-modal-dataset/rnc-modal-dataset.vue'

export default {
  name: 'RncModalBoth',
  components: {
    'rnc-modal-togglemask': RncModalTogglemask,
    'rnc-modal-model': RncModalModel,
    'rnc-modal-dataset': RncModalDataset,
  },
  data: function () {
    return {
      isAddModelShown: true,
    }
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

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

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
      cursor: pointer;
    }
    .selected {
      color: $black;
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
