<template>
  <transition
    v-if="showMenu"
    name="fade"
  >
    <div id="slide-menu">
      <div id="menu-list">
        <div
          v-for="(menu, key) in menuObj"
          :key="key"
          :class="'RncDrawerMenu-button' + key"
          class="task-button"
          @click="onItemClick(menu.title)"
        >
          <i
            :class="menu.icon"
            class="fa"
            aria-hidden="true"
          />
          {{ menu.title }}
        </div>
        <slot />
      </div>
      <div
        id="slide-mask"
        @click="$emit('show-drawer-menu')"
      />
    </div>
  </transition>
</template>

<script>
import { PAGE_ID } from '../../../const.js'

export default {
  name: 'RncDrawerMenu',
  props: {
    menuObj: {
      type: Array,
      default: _ => [],
    },
    showMenu: {
      type: Boolean,
      default: false
    }
  },
  data: function () {
    return {
      routerData: {
        train: { path: '/', page: PAGE_ID.TRAIN },
        predict: { path: '/predict', page: PAGE_ID.PREDICT },
        dataset: { path: '/dataset', page: PAGE_ID.DATASET },
        debug: { path: '/debug', page: PAGE_ID.DEBUG }
      }
    }
  },
  methods: {
    onItemClick: function (page_name) {
      const pageData = this.routerData[page_name.toLowerCase()]

      if (pageData == null) {
        throw new Error(page_name + ' is not supported page name.')
      }

      const path = pageData.path
      const page_number = pageData.page

      if (this.$router != null) {
        this.$router.push({ path })
      } else {
        console.warn(`The '$router' is not present in the Component. The path is: ${path}, and the page is: ${page_number}`)
      }

      this.$emit('show-drawer-menu', page_number)

      return path
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#slide-menu {
  position: fixed;
  z-index: 99;
  left: 0;
  top: $header-height;
  height: calc(100vh - #{$header-height});
  width: 100%;
  display: flex;

  #slide-mask {
    height: 100vh;
    width: 100%;
    background-color: $modal-mask-color;
  }

  #menu-list {
    height: 100vh;
    width: 10%;
    min-width: $slide-window-width-min;
    background-color: $dark-blue;

    .task-button {
      font-size: 125%;
      width: calc(100% - 30%);
      color: white;
      text-align: left;
      margin-top: 20%;
      margin-left: 20%;
    }
    .task-button:hover {
      color: gray;
      cursor: pointer;
    }
  }
}
</style>
