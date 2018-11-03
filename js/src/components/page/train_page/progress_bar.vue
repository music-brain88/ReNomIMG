<template>
  <div id="progress-bar">
    <div id="model-id-area">{{ this.model_id }}</div>
    <div id="epoch-area">{{ this.current_epoch }} / {{ this.total_epoch }}</div>
    <div id="batch-area">{{ this.current_batch }} / {{ this.total_batch }}</div>
    <div id="loss-area">{{ this.loss }}</div>
    <div id="bar-area">
      <div id="bar-background">
      </div>
    </div>
    <div id="button-stop-area"></div>
  </div>
</template>

<script>
export default {
  name: 'ProgressBar',
  props: {
    model: Object
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
        return model.id
      }
    },
    current_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return model.last_epoch
      }
    },
    current_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return model.last_batch
      }
    },
    total_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return model.total_epoch
      }
    },
    total_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return model.total_batch
      }
    },
    loss: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return model.train_loss[-1]
      }
    }
  },
  created: function() {
  },
  methods: {
  },
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
  align: center;

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
