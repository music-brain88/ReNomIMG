<template>
  <div id="app">
    <app-header></app-header>

    <div id="navigation">
      <transition name="mask">
        <div id='mask' v-if="isShown" @click='hideNavigationBar'></div>
      </transition>

      <transition name="nav">
        <navigation-bar v-if="isShown"></navigation-bar>
      </transition>
    </div>

    <div class="container">
      <router-view></router-view>
    </div>
  </div>
</template>

<script>
import AppHeader from './common/app_header.vue'
import NavigationBar from './common/navigation_bar.vue'

export default {
  name: "App",
  components:{
    'app-header': AppHeader,
    'navigation-bar': NavigationBar
  },
  data: function () {
    return {
      showNavigationBarFlag: true
    }
  },
  computed: {
    isShown: function () {
      return this.$store.getters.getNavigationBarShowFlag
    }
  },
  created: function() {
    this.$store.commit("setPageName", {
      "page_name": this.$router.currentRoute.name,
    });
  },
  methods: {
    hideNavigationBar: function () {
      this.$store.commit('setNavigationBarShowFlag', {
        flag: false,
      })
    }
  }
}
</script>

<style lang="scss">
  @import url('https://rsms.me/inter/inter-ui.css');
  * {
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
    -o-box-sizing: border-box;
    -ms-box-sizing: border-box;
    box-sizing: border-box;

    font-family: 'Inter UI', sans-serif;
  }
  #app {
    $max-width: 1280px;
    $header-height: 44px;

    position: relative;
    width: 100vw;
    height: calc(100% - #{$header-height});
    margin: 0 auto;
    padding: 0;
    padding-top: $header-height;

    .container {
      max-width: $max-width;
    }

    #navigation {
      position: fixed;
      z-index: 999;

      #navigation-bar {
        position: fixed;
        top: $header-height;
        z-index: 2;
      }
      #mask {
        position: fixed;
        top: $header-height;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 1;
        background-color: rgba(0, 0, 0, 0.4);
      }
    }

    .nav-enter-active {
      animation: slide-in 0.4s ease;
    }
    .nav-leave-active {
      animation: slide-in 0.2s ease reverse;
    }

    @keyframes slide-in {
      0% {
        left: -180px;
      }
      100% {
        left: 0;
      }
    }

    .mask-enter-active, .mask-leave-active {
      transition: all 0.3s;
    }
    .mask-enter-to, .mask-leave {
      opacity: 1;
    }
    .mask-enter, .mask-leave-to {
      opacity: 0;
    }
  }
</style>
