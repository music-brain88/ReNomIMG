<template>
  <div id="class-info-area">
    <!-- <div id="model-id-area">
      <span v-if="isTitle">Model</span>
      <span v-else>{{ this.model_id }}</span>
    </div>
    <div id="epoch-area">
      <span v-if="isTitle">Epoch</span>
      <span v-else>{{ this.current_epoch }} / {{ this.total_epoch }}</span>
    </div>
    <div id="batch-area">
      <span v-if="isTitle">Batch</span>
      <span v-else>{{ this.current_batch }} / {{ this.total_batch }}</span>
    </div>
    <div id="loss-area">
      <span v-if="isTitle">Loss</span>
      <span v-else-if="model.isTraining()">{{ this.loss }}</span>
      <span v-else-if="model.isValidating()">Validating</span>
      <span v-else-if="model.isStopping()">Stopping</span>
      <span v-else-if="model.isWeightDownloading()">Weight Downloading</span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle"></span>
      <div id="bar-background" v-else>
        <div id="bar-front"
          :style="getWidthOfBar"
          :class="[getColorClass(model), getBarClass]">
        </div>
      </div>
    </div>
    <div id="button-stop-area" v-if="!isTitle">
      <i class="fa fa-stop-circle-o" aria-hidden="true" @click="onStop"></i>
    </div> -->
    
    <div class="item-name">
      {{item_name}}
    </div>
    <div class="item-value"> 
      <div class="bar" :style="'width:' + calc_width(item_class_ratio) + '%;'">
        <section class="color-train" :style="'width:' + calc_width(item_train_ratio) + '%;background:#0762AD;'">
        </section>
        <section class="color-valid" :style="'width:' + calc_width(item_valid_ratio) + '%;background:#EF8200; '">
        </section>
        <section v-if="item_test_ratio" :style="'width:' + calc_width(item_test_ratio) + '%;background:red'">
        </section>
      </div>
    </div>

  </div>
</template>

<script>
import {RUNNING_STATE} from '@/const.js'
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'
export default {
  name: 'ProgressBar',
  props: {
    item_name: String,
    item_class_ratio: Number,
    item_test_ratio: Number,
    item_train_ratio: Number,
    item_valid_ratio: Number,
    height: Number,
    width: Number
  },
  data: function () {
    return {

    }
  },
  computed: {
    ...mapGetters([
      'getColorClass'
    ]),
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    getBarClass: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return 'validating'
      } else {
        return 'training'
      }
    },
    getWidthOfBar: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return {
          width: '20%'
        }
      } else {
        if (this.total_batch === 0) {
          return {
            width: 0 + '%'
          }
        } else {
          return {
            width: (this.current_batch / this.total_batch) * 100 + '%'
          }
        }
      }
    },
    current_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_epoch
      }
    },
    current_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_batch
      }
    },
    total_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_epoch
      }
    },
    total_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_batch
      }
    },
    loss: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.last_batch_loss.toFixed(3)
      }
    }
  },
  created: function () {

  },
  methods: {
    calc_width: function (width) {
      let percent = width * 100
      return percent
    }
  }
}
</script>

<style lang='scss' scoped>
#class-info-area {
  display: flex;
  flex-direction: row;
  justify-content: flex-start;
  align-items: center;
  width: 100%;
  height: calc(#{$progress-bar-height}*0.8);
  padding-left: $progress-bar-margin;
  padding-right: $progress-bar-margin;
  margin-bottom: 10px;
  font-size: 80%;
  text-align: center;

  .item-name {
    width: 30%;
  }

  .item-value {
    display: flex;
    width: 70%;
  }
  .bar {
    display: flex;
    justify-content: flex-start;
    width: 100%;
    height: 10px;
    overflow: hidden;
  }
  .bar section {
  	min-width: 10%;
    height: 10px;
    line-height: 10px;
    text-align: center;
    color: white;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    flex-wrap: nowrap;
  }
  
}
</style>
