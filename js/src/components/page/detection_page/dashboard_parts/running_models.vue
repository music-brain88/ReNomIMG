<template>
  <div id="running-models">
    <div class="title">
      Running Models
    </div>

    <div class="progress-bar-area">
      <model-progress v-for="(s, index) in runningModels" :key="index" :index="index" :model="s" v-if="s"></model-progress>
    </div>
  </div>
</template>

<script>
import { STATE_ID } from '@/constant'
import ModelProgress from './model_progress.vue'

export default {
  name: 'RunningModels',
  components: {
    'model-progress': ModelProgress
  },
  computed: {
    runningModels () {
      return this.$store.getters.getModelsFromState(STATE_ID['Running'])
    }
  }
}
</script>

<style lang="scss" scoped>
#running-models {

  width: 100%;
  height: 100%;
  margin-top: 24px;

  .title {
    font-family:$content-inner-header-font-family;
    font-size: $content-inner-header-font-size;
  }

  .progress-bar-area {
    height: calc(100% - #{$content-top-header-hight});
    overflow: auto;
  }
}
</style>
