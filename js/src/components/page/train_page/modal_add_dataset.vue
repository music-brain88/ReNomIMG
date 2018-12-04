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
        <select v-model="test_dataset" :disabled="isTestDataset">
          <option disabled value="" selected>Select Test Dataset</option>
          <option value="none">--none--</option>
          <option :value="item" v-for="item in getFilteredTestDatasetList">{{item.name}}</option>
        </select>
      </div>
      <div class='item'>
        Description<textarea type="text" v-model="description" placeholder="description"/>
      </div>
      <div class='item'>
        Ratio<input type="number" v-model="ratio" placeholder="0.8" step="0.1" min="0" max="1"/>
      </div>
      <input type="button" value="Confirm" @click="onAddDataset" :disabled="isComfirmable">
    </div>
    <div id="dataset-confirm">
      <div id="title">
        Dataset BreakDown
      </div>
      <div id="dataset-name">
        AA
      </div>
      <div id="dataset-taio">
        0.9
      </div>
      <div id="dataset-numbers">
        <div id="total-image-num">
          Total Images: 12000
        </div>
        <div id="train-image-num" class="num">
          Train: 10000
        </div>
        <div id="valid-image-num" class="nun">
          Valid: 1000
        </div>
        <div id="test-image-num" class="num">
          Test: 1000
        </div>
      </div>
      <div id="breakdown">
        Break Downs
      </div>
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
      test_dataset: '',
      timeStamp: ''
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
    ...mapActions([
      'createDataset',
      'createTestDataset',
      'confirmDataset',
      'confirmTestDataset'
    ]),
    onConfirm: function () {
      const date = new Date()
      this.timeStamp = date.getTime()
    },
    onAddDataset: function () {
      if (this.isTestDataset) {
        // this.createTestDataset({
        //   'name': this.name,
        //   'ratio': this.ratio,
        //   'description': this.description,
        // })
        this.confirmTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
        })
      } else {
        const test_dataset_id = this.test_dataset.id
        this.createDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': test_dataset_id,
        })
      }
    } // onAddDataset
  }
}
</script>

<style lang='scss'>
#modal-add-dataset {
  display: flex;
  width: 100%;
  height: 100%;
  padding: 10px;

  #title {
    color: gray;
  }
  #dataset-setting {
    height: 100%;
    width: 50%;
    font-size: 90%;
    .item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: calc(100% - 8%);
      margin: 4%;
      input, textarea {
        width: 50%;
      }
      select {
        background-color: white;
      }
    }
  }
  #dataset-confirm {
    width: 50%;
    height: 100%;
    #dataset-numbers {
      width: 100%;
      display: flex;
      justify-content: space-around;
      align-items: center;
      margin: 4%;
      #total-image-num {
        width: 31%;
      }
      .num {
        display: flex;
        justify-content: center;
        width: 23%;
      }
    }
  }
}
</style>
