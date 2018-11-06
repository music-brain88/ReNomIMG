<template>
  <transition name="fade">
  <div id='slide-menu' v-if='getShowSlideMenu'>
      <div id='menu-list'>
        <div id='button-classification' class='task-button'
          @click="setCurrentTask(TASK.CLASSIFICATION)">
          Classification
        </div>
        <div id='button-detection' class='task-button'
          @click="setCurrentTask(TASK.DETECTION)">
          Detection
        </div>
        <div id='button-segmentation' class='task-button'
          @click="setCurrentTask(TASK.SEGMENTATION)">
          Segmentation
        </div>
      </div>
      <div id='slide-mask' @click="showSlideMenu(false)" >
      </div>
    </div>
  </transition>
</template>

<script>
import {mapGetters, mapMutations} from 'vuex'
import C from '@/const.js'

export default {
  name: 'SlideMenu',
  computed: {
    ...mapGetters(['getShowSlideMenu']),
    TASK: function () {
      return C.TASK
    }
  },
  created: function () {
  },
  methods: {
    ...mapMutations(['showSlideMenu', 'setCurrentTask']),
  }
}
</script>

<style lang='scss'>
#slide-menu {
  position: fixed;
  left: 0;
  top: 0;
  height: 100vh;
  width: 100%;
  display: flex;

  #slide-mask {
    height: 100vh;
    width: 100%;
    background-color: $slide-window-mask-color;
  }

  #menu-list {
    height: 100vh;
    width: 10%;
    min-width: $slide-window-width-min;
    background-color: $slide-window-background-color;

    .task-button {
      width: 100%;
      height: 10%;
      // background-color: black;
      text-align: center;
    }
  }
}

</style>
