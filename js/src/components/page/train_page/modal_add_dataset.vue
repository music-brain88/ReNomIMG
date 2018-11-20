<template>
  <div id="modal-add-dataset">
    <div id="dataset-setting">
      <div id="title" v-if="isTestDataset"> Test Dataset Setting
      </div>
      <div id="title" v-else> Dataset Setting
      </div>
      <div class='item'>
        Create as Test Dataset
        <input type="checkbox" value="false" v-model="isTestDataset" placeholder="false">
      </div>
      <div class='item'>
        Name<input type="text" v-model="name" placeholder="dataset"/>
      </div>
      <div class='item'>
        Test Dataset
        <select v-model="test_dataset_id" :disabled="isTestDataset">
          <option disabled value="" selected>Select Test Dataset</option>
          <option value="">--none--</option>
          <option v-for="item in getFilteredTestDatasetList">{{item.name}}</option>
        </select>
      </div>
      <div class='item'>
        Description<textarea type="text" v-model="description" placeholder="description"/>
      </div>
      <div class='item'>
        Ratio<input type="number" v-model="ratio" placeholder="0.8"/>
      </div>
      <input type="button" value="Confirm" @click="onAddDataset" :disabled="isComfirmable">
    </div>
    <div id="dataset-confirm">
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
      description: '',
      ratio: 0.8,
      isTestDataset: false,
      test_dataset_id: ''
    }
  },
  computed: {
    ...mapGetters(['getFilteredTestDatasetList']),
    isComfirmable: function () {
      if (this.name && this.ratio > 0 && this.ratio < 1) {
        return false
      }
      return true
    }
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
          'description': this.description,
        })
      } else {
        this.createDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': this.test_dataset_id,
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
    width: 50%;
    font-size: 90%;
    #title {
      color: gray;
    }
    .item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: calc(100% - 8%);
      margin: 4%;
      input, textarea {
        width: 50%;
      }
    }
  }
}
</style>
