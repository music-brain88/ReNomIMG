<template>
  <div id="modal-add-dataset">
    <div id="dataset-setting">
      <div class='item'>
        Name: <input type="text" v-model="name" placeholder="dataset"/>
      </div>
      <div class='item'>
        Ratio: <input type="number" v-model="ratio" placeholder="0.8"/>
      </div>
      <div class='item'>
        Test Dataset: <input type="checkbox" value="false" v-model="isTestDataset" placeholder="false">
      </div>
      <input type="button" value="Confirm" @click="onAddDataset">
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'

export default {
  name: 'ModalAddDataset',
  components: {
  },
  data: function () {
    return {
      name: '',
      ratio: 0.8,
      isTestDataset: false,
    }
  },
  computed: {

  },
  created: function () {

  },
  methods: {
    ...mapActions(['createDataset', 'createTestDataset']),
    onAddDataset: function () {
      if (this.isTestDataset) {
        this.createTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': ' ',
        })
      } else {
        this.createDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': ' ',
          'test_dataset_id': 1,
        })
      }
    }
  }
}
</script>

<style lang='scss'>
#modal-add-dataset {
  display: flex;
  width: 100%;
  height: 100%;
  padding: 10px;
  #dataset-setting {
    height: 100%;
    width: 100%;
    .item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 50%;
      margin: 4%;
    }
  }
}
</style>
