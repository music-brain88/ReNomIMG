<template>
  <div id="modal-add-dataset">
    <!--Left page of creating dataset---------->
    <div id="dataset-setting">
      <div class="title">
        Dataset Setting
      </div>
      <div class="item-set-form">
        <div class="item-set">
          <div class="item">
            Name
            <rnc-input
              :input-max-length="nameMaxLength"
              :input-min-length="nameMinLength"
              :disabled="submitable"
              v-model="nameText"
              class="input-value"
              place-holder="dataset(Required)"
              @rnc-input="onUpdateName"
            />
          </div>
          <div
            v-if = "showNameErrorMessage"
            class="vali-mes"
          >
            {{ vali_params_name }}
          </div>
        </div>
        <div class="item-set">
          <div class="item">
            Description
            <rnc-input
              ref="descriptionText"
              :is-textarea="true"
              :input-max-length="descriptionMaxLength"
              :place-holder="'description(Optional)'"
              :disabled="submitable"
              :rows="3"
              v-model="descriptionText"
              class="input-value"
              @rnc-input="onUpdateDescription"
            />
          </div>
          <div
            v-if = "showDescriptionErrorMessage"
            class="vali-mes"
          >
            {{ vali_params_description }}
          </div>
        </div>
        <div class="item-set">
          <div class="item">
            Ratio
            <rnc-input
              :input-min-length="3"
              :input-max-length="4"
              :input-min-value="0.3"
              :input-max-value="0.99"
              :only-number="true"
              :disabled="submitable"
              v-model="ratio"
              class="input-value"
              step="0.1"
              @rnc-input="onUpdateRatio"
            />
          </div>
          <div
            v-if = "showRatioErrorMessage"
            class="vali-mes"
          >
            {{ vali_params_ratio }}
          </div>
        </div>
      </div>
      <rnc-button
        :disabled="!confirmable || submitable"
        class="confirm-button"
        button-label="Confirm"
        @click="onConfirmDataset"
      />
    </div>
    <!----------Left page of creating dataset-->

    <!--Right page of creating dataset---------->
    <div id="dataset-confirm">
      <div class="title">
        Dataset BreakDown
      </div>
      <div class="param-set-form">
        <div class="dataset-params">
          <span>
            Name : {{ nameText }}
          </span>
        </div>
        <div class="dataset-params">
          <span>
            Ratio : {{ ratio }}
          </span>
        </div>
        <div class="dataset-numbers">
          <div class="num-item">
            <div class="num-title">
              Total Images :
            </div>
            <div class="num">
              {{ total_num }}
            </div>
          </div>
        </div>
        <rnc-bar-dataset
          id="dataset-ratio-bar"
          :class="{'bar-anime': confirming_dataset}"
          :train-num="train_num"
          :valid-num="valid_num"
        />
        <!-- <div
          id="dataset-ratio-bar"
          :class="{'bar-anime': confirming_dataset}"
          @mouseenter="isHovering=true"
          @mouseleave="isHovering=false"
        >
          <section
            :style="train_num_style"
            class="color-train"
          >
            <span>
              Train
            </span>
          </section>
          <section
            :style="valid_num_style"
            class="color-valid"
          >
            <span>
              Valid
            </span>
          </section>
        </div> -->
        <div id="breakdown">
          <div
            v-if="confirming_flag"
            id="load-progress"
          >
            <div class="lds-ripple">
              <div />
              <div />
            </div>
            Loading Dataset...
          </div>
          <div
            v-for="(item, key) in class_items"
            v-else
            id="class-ratio-bars"
            :key="key"

          >
          <div
            id="dataset-class-bars"
            :class="{'bar-anime': bar_move}"
          >
            <div class="bar">
                <rnc-bar-dataset
                  :train-num="train_num"
                  :valid-num="valid_num"
                  :class-name="item[0]"
                  :class-ratio="item[1]"
                />
            </div>
          </div>
            </div>
          <!-- <div
            v-for="(item, key) in class_items"
            v-else
            id="class-ratio-bars"
            :key="key"
          >
            <span>
              {{ item[0] }}
            </span>
            <div
              :class="{'bar-anime': confirming_dataset}"
              :style="{width: item[1] + item[2] + '%'}"
              class="bar"
            >
              <section
                :style="{width: item[1]/(item[1] + item[2])*100 + '%'}"
                class="color-train"
              />
              <section
                :style="{width: item[2]/(item[1] + item[2])*100 + '%'}"
                class="color-valid"
              />
            </div>
          </div> -->
        </div>
      </div>
      <rnc-button
        :disabled="!submitable"
        button-label="Submit"
        class="submit-button"
        @click="onAddDataset"
      />
      <rnc-button
        :disabled="!submitable"
        :cancel="cancel"
        button-label="Back"
        @click="backDataset"
      />
    </div>
    <!----------Right page of creating dataset-->
  </div>
</template>

<script>
import { mapState, mapGetters, mapMutations, mapActions } from 'vuex'
import { DATASET_NAME_MAX_LENGTH, DATASET_NAME_MIN_LENGTH,
  DATASET_DESCRIPTION_MAX_LENGTH, DATASET_DESCRIPTION_MIN_LENGTH } from './../../../const.js'

import RncButton from '../../Atoms/rnc-button/rnc-button.vue'
import RncSelect from '../../Atoms/rnc-select/rnc-select.vue'
import RncInput from '../../Atoms/rnc-input/rnc-input.vue'
import RncBarDataset from '../../Atoms/rnc-bar-dataset/rnc-bar-dataset'

export default {
  name: 'ModalAddDataset',
  components: {
    'rnc-button': RncButton,
    'rnc-select': RncSelect,
    'rnc-input': RncInput,
    'rnc-bar-dataset': RncBarDataset
  },
  data: function () {
    return {
      // name: '',
      nameText: '',
      descriptionText: '',
      ratio: 0.8,
      test_dataset: '',
      timeStamp: '',
      isHovering: false,
      notifyDescriptionField: false,
      nameFieldTimeoutFunc: function () {},
      paramKeys: {},
      vali_params_name: '',
      vali_params_description: '',
      vali_params_ratio: '',
      showNameErrorMessage: false,
      showDescriptionErrorMessage: false,
      showRatioErrorMessage: false,
      cancel: true
    }
  },
  computed: {
    ...mapState([
      'confirming_flag',
      'confirming_dataset'
    ]),
    ...mapGetters([
      'getFilteredTestDatasetList',
    ]),
    nameMaxLength: function () { return DATASET_NAME_MAX_LENGTH },
    nameMinLength: function () { return DATASET_NAME_MIN_LENGTH },
    descriptionMaxLength: function () { return DATASET_DESCRIPTION_MAX_LENGTH },
    descriptionMinLength: function () { return DATASET_DESCRIPTION_MIN_LENGTH },
    confirmable: function () {
      if (!this.nameText || this.confirming_flag || this.vali_params_name || this.vali_params_ratio || this.vali_params_description) {
        return false
      }
      return true
    },
    submitable: function () {
      // TODO: console.log('*** this.confirming_flag ***' + this.confirming_flag)
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
    },
  },
  // TODO: watch: {
  // TODO:   confirming_flag: function () {
  // TODO:     console.log('*** this.confirming_flag ***' + this.confirming_flag)
  // TODO:     console.log('*** this.confirming_dataset ***' + this.confirming_dataset)
  // TODO:     console.dir(this.confirming_dataset)
  // TODO:   }
  // TODO: },
  beforeMount: function () {
    this.reset()
  },
  beforeDestroy: function () {
    this.backDataset()
  },

  methods: {
    ...mapMutations([
      'setConfirmingFlag',
      'setConfirmingDataset'
    ]),
    ...mapActions([
      'createDataset',
      // TODO: 'createTestDataset',
      'confirmDataset',
      // TODO: 'confirmTestDataset',
      'deleteDataset'
    ]),
    onUpdateName: function (params) {
      this.vali_params_name = params['errorMessage']
      this.showNameErrorMessage = true
    },

    onUpdateDescription: function (params) {
      this.vali_params_description = params['errorMessage']
      this.showDescriptionErrorMessage = true
    },

    onUpdateRatio: function (params) {
      this.showRatioErrorMessage = true
      this.vali_params_ratio = params['errorMessage']
    },

    descriptionInputNotify: function (e) {
      this.notifyDescriptionField = (this.description.length === this.descriptionMaxLength)
    },
    onConfirmDataset: function () {
      const date = new Date()
      this.timeStamp = date.getTime()
      this.setConfirmingFlag(true)
      if (this.isTestDataset) {
        // TODO: this.confirmTestDataset({
        // TODO:   'name': this.nameText,
        // TODO:   'ratio': this.ratio,
        // TODO:   'description': this.descriptionText,
        // TODO: })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.confirmDataset({
          'hash': this.timeStamp,
          'name': this.nameText,
          'ratio': this.ratio,
          'description': this.descriptionText,
          'test_dataset_id': test_dataset_id,
        })
      }
    },
    onAddDataset: function () {
      if (this.isTestDataset) {
        // TODO: this.createTestDataset({
        // TODO:   'name': this.nameText,
        // TODO:   'ratio': this.ratio,
        // TODO:   'description': this.descriptionText,
        // TODO: }).then(() => {
        // TODO:   this.reset()
        // TODO: })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.createDataset({
          // TODO: 'hash': this.timeStamp,
          // TODO: 'name': this.nameText,
          // TODO: 'ratio': this.ratio,
          // TODO: 'description': this.descriptionText,
          // TODO: 'test_dataset_id': test_dataset_id,
          'dataset_id': this.confirming_dataset.id
        }).then(() => {
          this.reset()
          this.$parent.showAddModel()
        })
      }
    },
    backDataset: function () {
      if (this.confirming_dataset) {
        this.deleteDataset(this.confirming_dataset.id)
      }
    },

    reset: function () {
      this.nameText = ''
      this.descriptionText = ''
      this.ratio = 0.8
      this.timeStamp = ''
      this.setConfirmingDataset(null)
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

.confirm-button {
  margin-left: 72%;
}
#modal-add-dataset {
  display: flex;
  width: 100%;
  height: 100%;
  padding: 10px;
  font-size: $component-font-size-small;
  .title {
    width: 100%;
    height: 5%;
    color: gray;
    font-size: $component-font-size;
    align-items: center;
    justify-content: space-between;
  }
  #dataset-setting {
    height: calc(100% - 10px);
    width: 50%;
    position: relative;
    .item-set-form {
      overflow: auto;
      height: calc(100% - 10%);
      margin-top: 5px;
      margin-bottom: 5px;
      .item-set {
        margin: $margin-middle;
        .item {
          height: 10%;
          color: $component-font-color;
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: calc(100% - $margin-middle);
          .input-value {
            width: 50%;
          }
        }
        .vali-mes {
          color: $err_red;
          font-size: $fs-small;
          text-align: right;
          margin-top: $margin-micro;
        }
      }
    }
  }
  #dataset-confirm {
    color: $component-font-color;
    width: 50%;
    height: calc(100% - 10px);
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    .param-set-form {
      overflow: visible;
      height: calc(100% - 10%);
      width: 100%;
      margin-top: 5px;
      margin-bottom: 5px;
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
      .dataset-params {
        width: calc(100% - $margin-middle);
        height: 10%;
        display: flex;
        margin: $margin-middle;
        span {
          width: calc(55%);
        }
      }
      .dataset-numbers {
        width: calc(100% - $margin-middle);
        display: flex;
        margin: 16px 16px 3px 16px;

        .num-item {
          width: 100%;
          display: flex;
          .num-title {
          }
          .num {
            margin-left: 2%;
          }
        }
      }
      #breakdown {
        width: calc(100% - 5.281);
        margin-top: 2%;
        height: calc(100% - 5% - 3% - 1.6rem - 3% - 1.6rem - 6% - 20px - 2% - 40px - 2%);
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
          #dataset-class-bars {
            width: 100%;
          }
          .bar {
            height: 16px;
            width: 94%;
            display: flex;
          }
          section {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
          }
        }
        #load-progress {
          width: 100%;
          height: 100%;
          display: flex;
          padding-bottom: 10%;
          align-items: center;
          justify-content: center;
          .lds-ripple {
            display: inline-block;
            position: relative;
            width: 64px;
            height: 64px;
            top: 3px;
            left: 3px;
          }
          .lds-ripple div {
            position: absolute;
            border: 4px solid #777;
            opacity: 1;
            border-radius: 50%;
            animation: lds-ripple 1s cubic-bezier(0, 0.2, 0.8, 1) infinite;
            animation-fill-mode: both;
          }
          .lds-ripple div:nth-child(2) {
            animation-delay: -0.5s;
            animation-fill-mode: both;
          }
          @keyframes lds-ripple {
            0% {
              top: 28px;
              left: 28px;
              width: 0;
              height: 0;
              opacity: 1;
            }
            100% {
              top: -1px;
              left: -1px;
              width: 58px;
              height: 58px;
              opacity: 0;
            }
          }
        }
      }
    }
    .submit-button {
      margin-right: $margin-micro;
    }
  }

  .color-train {
    background-color: #0762AD;
  }
  .color-valid {
    background-color: #EF8200;
  }
  .train-anime {
    animation: growXTrain 0.8s linear;
    animation-fill-mode: both;
  }
  .valid-anime {
    animation: growXValid 0.8s linear;
    animation-fill-mode: both;
  }
  .bar-anime {
    animation: growX 0.8s;
    animation-fill-mode: both;
    animation-iteration-count: 1;
  }
  @keyframes growX {
    0% {
      transform: translateX(-50%) scaleX(0);
    }
    100% {
      transform: translateX(0) scaleX(1);
    }
  }
  .show-short-period {
    animation: notifyAnimation ease-in 3s;
  }
  @keyframes notifyAnimation {
   0% {
      outline-color: red;
    }
   80% {
      outline-color: red;
   }
  }
}
</style>
