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
      <input type="button" value="Confirm" @click="onConfirmDataset" :disabled="isComfirmable">
    </div>
    <div id="dataset-confirm">
      <div id="title">
        Dataset BreakDown
      </div>
      <div id="dataset-name">
        <span v-if="getDatasetDetail"> {{getDatasetDetail.dataset_name}} </span>
      </div>
      <div id="dataset-ratio">
        <span v-if="getDatasetDetail"> {{getDatasetDetail.ratio}} </span>
      </div>
      <div id="dataset-numbers">
        <div id="total-image-num">
          Total Images: <span v-if="getDatasetDetail">{{getDatasetDetail.train_data.img.length + getDatasetDetail.valid_data.img.length}}</span>
        </div>
        <div id="train-image-num" class="num">
          Train: <span v-if="getDatasetDetail">{{getDatasetDetail.train_data.img.length}}</span> 
        </div>
        <div id="valid-image-num" class="num">
          Valid: <span v-if="getDatasetDetail">{{getDatasetDetail.valid_data.img.length}} </span>
        </div>
        <div id="test-image-num" class="num">
          Test: 
        </div>
      </div>
      <div id="dataset-ratio-bar">
        <dataset-ratio-bar
          :item_train_ratio="getDatasetDetail.train_data.img.length/(getDatasetDetail.train_data.img.length + getDatasetDetail.valid_data.img.length)"
          :item_valid_ratio="getDatasetDetail.valid_data.img.length/(getDatasetDetail.train_data.img.length + getDatasetDetail.valid_data.img.length)"
        >
        </dataset-ratio-bar>  
      </div>
      <div id="breakdown">
        Break Downs
        <div v-if="getDatasetDetail">
          <breakdown-ratio-bar 
            v-for="item in getDatasetDetail.class_info.class.length"
            :key="item"
            :item_name="getDatasetDetail.class_info.class[item - 1]"
            :item_class_ratio="getDatasetDetail.class_info.class_ratio[item - 1]"
            :item_test_ratio="getDatasetDetail.class_info.test_ratio[item - 1]"
            :item_train_ratio="getDatasetDetail.class_info.train_ratio[item - 1]"
            :item_valid_ratio="getDatasetDetail.class_info.valid_ratio[item - 1]"
            >
          </breakdown-ratio-bar>
          <input type="button" value="submit" @click="onAddDataset" :disabled="isComfirmable">
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'
import BreakDownBar from '@/components/page/train_page/breakdown_ratio_bar.vue'
import DatasetDetailBar from '@/components/page/train_page/dataset_detail_ratio_bar.vue'

export default {
  name: 'ModalAddDataset',
  components: {
    'dataset-ratio-bar': DatasetDetailBar,
    'breakdown-ratio-bar': BreakDownBar
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
    ...mapGetters([
      'getFilteredTestDatasetList',
      'getDatasetDetail'
    ]),
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
    onConfirmDataset: function () {
      if (this.isTestDataset) {
        this.confirmTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
        })
      } else {
        const test_dataset_id = this.test_dataset.id
        console.log('test', test_dataset_id)
        this.confirmDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': test_dataset_id,
        })
      }
    },
    onAddDataset: function () {
      if (this.isTestDataset) {
        this.createTestDataset({
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
    },
    hasData: function (data) {
      let value = data.length > 0 ? data : 'No Test Dataset Selected'
      return value
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
