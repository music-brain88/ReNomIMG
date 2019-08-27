<template>
  <header>
    <div
      id="app-header"
      @click="toggle"
    >
      <div id="menu-title">
        <div id="header-menu">
          <i
            class="fa fa-bars"
            aria-hidden="true"
            @click="showSlideMenu(!getShowSlideMenu)"
          />
        </div>
        <div id="title">
          <span
            id="product-title"
            class="header-title"
          >
            {{ ProductName }}
          </span>
          <span class="header-title">
            >
          </span>
          <span
            id="task-title"
            class="header-title"
          >
            {{ getCurrentTaskTitle }}
          </span>
          <span class="header-title">
            >
          </span>
          <span
            id="page-title"
            class="header-title"
          >
            {{ getCurrentPageTitle }}
          </span>
        </div>
      </div>
      <div id="task-buttons">
        <div
          id="classification-button"
          :class="{selectedTask: getCurrentTask === TASK.DETECTION}"
          @click="() => {setCurrentTask(TASK.DETECTION); init();}"
        >
          Detection
        </div>
        <div
          id="classification-button"
          :class="{selectedTask: getCurrentTask === TASK.SEGMENTATION}"
          @click="() => {setCurrentTask(TASK.SEGMENTATION); init();}"
        >
          Segmentation
        </div>
        <div
          id="classification-button"
          :class="{selectedTask: getCurrentTask === TASK.CLASSIFICATION}"
          @click="() => { setCurrentTask(TASK.CLASSIFICATION); init();}"
        >
          Classification
        </div>
      </div>
    </div>
  </header>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import { TASK_ID } from '../../../const.js'

export default {
  name: 'RncHeader',
  props: {
    ProductName: {
      type: String,
      default: 'ReNomIMG',
    }
  },
  computed: {
    ...mapGetters([
      'getCurrentTask',
      'getShowSlideMenu',
      'getCurrentTaskTitle',
      'getCurrentPageTitle']),
    TASK: function () {
      return TASK_ID
    }
  },
  created: function () {

  },
  methods: {
    ...mapActions(['init']),
    ...mapMutations(['showSlideMenu', 'setCurrentTask', 'showModal']),
    toggle: function () {
      this.showModal({ 'all': false })
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#app-header {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000;
  width: 100%;
  height: $header-height;
  min-height: $header-min-height;
  background-color: $dark-blue;
  color: $white;
  display: flex;
  justify-content: space-between;

  #task-buttons {
    width: 50%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    div {
      margin-right: $header-title-margin-left;
      cursor: pointer;
    }
    div:hover {
      color: gray;
    }
    .selectedTask {
      border-bottom: solid 1.5px white;
    }
    .selectedTask:hover {
      border-bottom: solid 1.5px gray;
    }
  }

  #menu-title {
    height: 100%;
    width: 50%;
    display: flex;
    #header-menu {
      width: 20px;
      height: 100%;
      margin-left: $header-title-margin-left;
      display: inline-flex;
      font-size: 100%;
      align-items: center;
    }

    #header-menu:hover {
      cursor: pointer;
      color: gray;
    }

    #title {
      height: 100%;
      margin: 0 0 0 $header-title-margin-left;

      .header-title {
        height: $header-height;
        min-height: $header-min-height;
        display: inline-flex;
        align-items: center;
      }

      #product-title {
        font-family: $header-product-name-font-family;
        font-size: $header-product-name-font-size;
      }

      #task-title {
        font-family: $header-title-font-family;
        font-size: $header-title-font-size;
        font-weight: bold;
      }

      #page-title {
        font-family: $header-title-font-family;
        font-size: $header-title-font-size;
      }
    }
  }
}
</style>
