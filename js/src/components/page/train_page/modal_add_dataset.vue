<template>
  <div id="modal-add-dataset">

    <!--Left page of creating dataset---------->
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
      <input type="button" value="Confirm" @click="onConfirmDataset" :disabled="!confirmable">
    </div>
    <!----------Left page of creating dataset-->
    
    <!--Right page of creating dataset---------->
    <div id="dataset-confirm">
      <div id="title">
        Dataset BreakDown
      </div>
      <div id="dataset-params">
        <span> Name : {{ name }} </span>
        <span> Ratio : {{ ratio }} </span>
      </div>
      <div id="dataset-numbers">
        <div class="num-item">
          <div class="num-title">
            Total Images :
          </div>
          <div class="num">
            {{ total_num }}
          </div>
        </div>
      </div>

      <div id="dataset-ratio-bar" @mouseenter="isHovering=true" @mouseleave="isHovering=false">
        <section class="color-train" :style="train_num_style">
          <transition name="fade">
            <span v-if="!isHovering">Train</span>
            <span v-else>{{train_num}}</span>
          </transition>
        </section>
        <section class="color-valid" :style="valid_num_style">
          <transition name="fade">
            <span v-if="!isHovering">Valid</span>
            <span v-else>{{valid_num}}</span>
          </transition>
        </section>
      </div>
      <div id="breakdown">
        <div id="class-ratio-bars" v-for="item in class_items"
          @mouseenter="isHovering=true" @mouseleave="isHovering=false">
          <span>{{item[0]}}</span>
          <section class="color-train" :style="{width: item[1] + '%'}"/>
          <section class="color-valid" :style="{width: item[2] + '%'}"/>
          <span v-if="isHovering">&nbsp;&nbsp;{{item[1].toFixed(2)}}-</span>
          <span v-if="isHovering">{{item[2].toFixed(2)}}[%]</span>
        </div>
      </div>
      <input id="submit-button" type="button"
        value="submit" @click="onAddDataset" :disabled="!submitable">
    </div>
    <!----------Right page of creating dataset-->
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
      timeStamp: '',
      isHovering: false
    }
  },
  beforeMount: function () {
    this.reset()
  },
  computed: {
    ...mapState([
      'confirming_flag',
      'confirming_dataset'
    ]),
    ...mapGetters([
      'getFilteredTestDatasetList',
    ]),
    confirmable: function () {
      if (!this.name || this.ratio <= 0 || this.ratio >= 1 || this.confirming_flag) {
        return false
      }
      return true
    },
    submitable: function () {
      if (!this.confirming_dataset || this.confirming_flag) {
        return false
      }
      return true
    },
    info: function () {
      const dataset = this.confirming_dataset
      if (!dataset) return
      const info = dataset.class_info
      return info
    },
    total_num: function () {
      const info = this.info
      if (!info) return
      return info.train_img_num + info.valid_img_num
    },
    train_num: function () {
      const info = this.info
      if (!info) return 0
      return info.train_img_num
    },
    valid_num: function () {
      const info = this.info
      if (!info) return 0
      return info.valid_img_num
    },
    class_items: function () {
      const dataset = this.confirming_dataset
      if (!dataset) return
      const class_map = dataset.class_map
      if (!this.info) return
      const train_list = this.info.train_ratio
      const valid_list = this.info.valid_ratio
      const class_list = this.info.class_ratio
      return train_list.map((t, index) => [
        class_map[index],
        (t) * class_list[index] * 100,
        valid_list[index] * class_list[index] * 100
      ])
    },
    train_num_style: function () {
      if (this.total_num <= 0) return
      return {
        width: (this.train_num / this.total_num) * 100 + '%',
      }
    },
    valid_num_style: function () {
      if (this.total_num <= 0) return
      return {
        width: (this.valid_num / this.total_num) * 100 + '%',
      }
    }
  },
  methods: {
    ...mapMutations([
      'setConfirmingFlag',
      'setConfirmingDataset'
    ]),
    ...mapActions([
      'createDataset',
      'createTestDataset',
      'confirmDataset',
      'confirmTestDataset'
    ]),
    onConfirmDataset: function () {
      const date = new Date()
      this.timeStamp = date.getTime()
      this.setConfirmingFlag(true)
      if (this.isTestDataset) {
        this.confirmTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
        })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.confirmDataset({
          'hash': this.timeStamp,
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
        }).then(() => {
          this.reset()
        })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.createDataset({
          'hash': this.timeStamp,
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': test_dataset_id,
        }).then(() => {
          this.reset()
        })
      }
    },
    reset: function () {
      this.name = ''
      this.description = ''
      this.ratio = 0.8
      this.isTestDataset = false
      this.test_dataset = ''
      this.timeStamp = ''
      this.setConfirmingDataset(null)
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
  font-size: $component-font-size-small;
  #title {
    width: 100%;
    height: 5%;
    color: gray;
    font-size: $component-font-size;
  }
  #dataset-setting {
    height: 100%;
    width: 50%;
    .item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: calc(100% - 8%);
      margin: 3%;
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
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    #dataset-ratio-bar {
      width: calc(100% - 30px);
      height: 20px;
      margin-bottom: 2%;
      display: flex;
      margin-left: 30px;
      section {
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
      }
    }
    #dataset-params {
      width: calc(100% - 3%);
      display: flex;
      margin-top: 3%;
      margin-left: 3%;
      span {
        width: calc(30%);
      }
    }
    #dataset-numbers {
      width: calc(100% - 3%);
      display: flex;
      margin-top: 3%;
      margin-left: 3%;
      margin-bottom: 3%;
      .num-item {
        width: 30%;
        display: flex;
        .num-title {
        }
        .num {
          margin-left: 3%;
        }
      }
    }
    #breakdown {
      width: 100%;
      margin-top: 2%;
      height: calc(100% - 5% - 3% - 1.6rem - 6% - 20px - 2% - 40px - 2%);
      overflow: auto;
      #class-ratio-bars {
        height: 18px;
        width: 100%;
        display: flex;
        span:nth-child(1) {
          width: 20%;
          display: flex;
          justify-content: flex-end;
          margin-right: 5px;
        }
        section {
          height: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }
      }
    }
    #submit-button {
    }
  }
  .color-train {
    background-color: #0762AD;
  }
  .color-valid {
    background-color: #EF8200;
  }
}
</style>
