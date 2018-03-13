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
import ModelProgress from './model_progress.vue'

export default {
  name: "RunningModels",
  components: {
    "model-progress": ModelProgress,
  },
  computed: {
    runningModels() {
      return this.$store.getters.getRunningModels;
    },
  },
}
</script>

<style lang="scss" scoped>
#running-models {
  $title-height: 24px;
  $title-font-size: 16px;
  $font-weight-medium: 500;

  width: 100%;
  height: 100%;
  margin-top: 24px;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .progress-bar-area {
    height: calc(100% - #{$title-height});
    overflow: auto;
  }
}
</style>
