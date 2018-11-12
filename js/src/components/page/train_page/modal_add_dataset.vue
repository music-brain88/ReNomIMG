<template>
  <div id="modal-add-dataset">
    Name: <input type="text" v-model="name" placeholder="dataset"/>
    Ratio: <input type="number" v-model="ratio" placeholder="0.8"/>
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
    }
  },
  computed: {

  },
  created: function () {

  },
  methods: {
    ...mapActions(['createDataset']),
    onAddDataset: function () {
      this.createDataset({
        'name': this.name,
        'ratio': this.ratio,
        'description': ' ',
        'test_dataset_id': 1,
      })
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
