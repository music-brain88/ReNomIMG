<template>
  <div id="progress-bar">
    <div id="model-id-area">
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
      <span v-else>{{ this.loss }}</span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle">Progress</span>
      <div id="bar-background" v-else>
      </div>
    </div>
    <div id="button-stop-area"></div>
  </div>
</template>

<script>
export default {
  name: 'ProgressBar',
  props: {
    model: Object,
    isTitle: {
      type: Boolean,
      default: false
    }
  },
  data: function () {
    return {
    }
  },
  computed: {
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    current_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.last_epoch
      }
    },
    current_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.last_batch
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
        return this.model.train_loss[-1]
      }
    }
  },
  created: function () {
  },
  methods: {
  }
}
</script>

<style lang='scss'>
#progress-bar {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: $progress-bar-height;
  padding-left: $progress-bar-margin;
  padding-right: $progress-bar-margin;
  text-align: center;

  #model-id-area {
    width: 12.5%;
    height: 100%;
  }
  #epoch-area {
    width: 12.5%;
    height: 100%;
  }
  #batch-area {
    width: 12.5%;
    height: 100%;
  }
  #loss-area {
    width: 20%;
    height: 100%;
  }
  #bar-area {
    width: 37.5%;
    height: 100%;
    padding-top: $bar-margin;
    padding-bottom: $bar-margin;
    #bar-background {
      width: 100%;
      height: calc(100% - #{$bar-margin});
      background-color: red; 
    }
  }
  #button-stop-area {
    width: 5%;
    height: 100%;
  }
}
</style>
