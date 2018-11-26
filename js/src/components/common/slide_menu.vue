<template>
  <transition name="fade">
  <div id='slide-menu' v-if='getShowSlideMenu'>
      <div id='menu-list'>
        <div class='task-button'
          @click="onItemClick('train')">
          Train
        </div>
        <div class='task-button'
          @click="onItemClick('predict')">
          Predict
        </div>
        <div class='task-button'
          @click="onItemClick('dataset')">
          Dataset
        </div>
      </div>
      <div id='slide-mask' @click="showSlideMenu(false)" >
      </div>
    </div>
  </transition>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import { TASK_ID } from '@/const.js'

export default {
  name: 'SlideMenu',
  computed: {
    ...mapGetters(['getShowSlideMenu']),
    TASK: function () {
      return TASK_ID
    }
  },
  created: function () {
    this.init()
  },
  methods: {
    ...mapMutations(['showSlideMenu', 'setCurrentTask']),
    ...mapActions(['init']),
    onItemClick: function (page_name) {
      this.init()
      if (page_name === 'train') {
        this.$router.push({path: '/'})
        this.setCurrentPage(PAGE_ID.TRAIN)
      } else if (page_name === 'predict') {
        this.$router.push({path: '/predict'})
        this.setCurrentPage(PAGE_ID.PREDICT)
      } else if (page_name === 'dataset') {
        this.$router.push({path: '/dataset'})
        this.setCurrentPage(PAGE_ID.DATASET)
      } else if (page_name === 'debug') {
        this.$router.push({path: '/debug'})
        this.setCurrentPage(PAGE_ID.DEBUG)
      } else {
        console.log(page_name + 'is not supported page name.')
      }
    }

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
