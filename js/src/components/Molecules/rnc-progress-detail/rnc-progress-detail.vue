<template>
  <div id="rnc-progress-detail">
    <div id="model-id-area">
      <span v-if="isTitle">
        Model
      </span>
      <span v-else>
        {{ model_id }}
      </span>
    </div>
    <div id="epoch-area">
      <span v-if="isTitle">
        Epoch
      </span>
      <span v-else>
        {{ current_epoch }} / {{ total_epoch }}
      </span>
    </div>
    <div id="batch-area">
      <span v-if="isTitle">
        Batch
      </span>
      <span v-else>
        {{ current_batch }} / {{ total_batch }}
      </span>
    </div>
    <div id="loss-area">
      <span v-if="isTitle">
        Loss
      </span>
      <span v-else-if="model">
        <span v-if="model.isTraining()">
          {{ loss }}
        </span>
        <span v-else-if="model.isValidating()">
          Validating
        </span>
        <span v-else-if="model.isStopping()">
          Stopping
        </span>
        <span v-else-if="model.isWeightDownloading()">
          Weight Downloading
        </span>
      </span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle" />
      <rnc-bar-progress
        v-else
        :bar-class="getBarClass"
        :color-class="colorClass"
        :total-batch="convert_string_to_zero(total_batch)"
        :current-batch="convert_string_to_zero(current_batch)"
      />
    </div>
    <rnc-button-stop
      v-if="!isTitle"
      :style="{width: '9%', height:'100%'}"
      @click="onStop"
    />
  </div>
</template>

<script>
import RncBarProgress from './../../Atoms/rnc-bar-progress/rnc-bar-progress.vue'
import RncButtonStop from './../../Atoms/rnc-button-stop/rnc-button-stop.vue'

export default {
  name: 'RncProgressDetail',
  components: {
    'rnc-bar-progress': RncBarProgress,
    'rnc-button-stop': RncButtonStop
  },
  props: {
    model: {
      type: Object,
      default: () => { undefined }
    },
    isTitle: {
      type: Boolean,
      default: false
    },
    colorClass: {
      type: String,
      default: 'color-0',
      validator: val => ['color-created', 'color-reserved', 'color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5'].includes(val)
    },
  },
  computed: {
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    getBarClass: function () {
      if (undefined === this.model) {
        return '-'
      }
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return 'validating'
      } else {
        return 'training'
      }
    },
    getWidthOfBar: function () {
      if (undefined === this.model) {
        return '-'
      }
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
    convert_string_to_zero: function (val) {
      const ret = (typeof val === 'string') ? 0 : val
      return ret
    },
    onStop: function () {
      if (this.model) {
        const model_id = this.model.id
        this.$emit('click-stop-button', model_id)

        // TODO : commit mutaion in the parent
        // const func = this.stopModelTrain
        // this.showConfirm({
        //   message: "Are you sure you want to <span style='color: #f00;}'>stop</span> this model?",
        //   callback: function () { func(id) }
        // })
      }
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#rnc-progress-detail {
  display: flex;
  flex-direction: row;
  align-items: center;
  width: 100%;
  height: $progress-bar-height;
  margin-bottom: $progress-bar-margin-bottom;
  font-size: $component-font-size-small;

  #model-id-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #epoch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #batch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 15%;
    height: 100%;
  }
  #loss-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18%;
    height: 100%;
  }

  #bar-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 33%;
    height: 70%;
  }
}
</style>
