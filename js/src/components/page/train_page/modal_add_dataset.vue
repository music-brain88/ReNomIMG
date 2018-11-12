<template>
  <div id="modal-add-dataset">
    Name: <input type="text" v-model="name" placeholder="dataset"/>
    Ratio: <input type="number" v-model="ratio" placeholder="0.8"/>
    Test Dataset: <input type="checkbox" value="false" v-model="isTestDataset" placeholder="false">
    <input type="button" value="Confirm" @click="onAddDataset">
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
  width: 100%;
  height: 100%;
}
</style>
