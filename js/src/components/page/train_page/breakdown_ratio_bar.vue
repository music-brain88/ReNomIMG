<template>
  <div id="class-info-area">
    <div class="item-name">
      {{ item_name }}
    </div>
    <div class="item-value">
      <div
        :style="'width:' + calc_width(item_class_ratio) + '%;'"
        class="bar">
        <section
          :style="'width:' + calc_width(item_train_ratio) + '%;background:#0762AD;'"
          class="color-train"/>
        <section
          :style="'width:' + calc_width(item_valid_ratio) + '%;background:#EF8200; '"
          class="color-valid"/>
        <section
          v-if="item_test_ratio"
          :style="'width:' + calc_width(item_test_ratio) + '%;background:red'"/>
      </div>
    </div>

  </div>
</template>

<script>
import { RUNNING_STATE } from '@/const.js'
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
      const percent = width * 100
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
