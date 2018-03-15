<template>
<div id="navigation">
  <transition name="mask">
    <div id='mask' v-if="$store.state.navigation_bar_shown_flag" @click='hideMenu'></div>
  </transition>

  <transition name="nav">
    <div id="navigation-bar" v-if="$store.state.navigation_bar_shown_flag">
      <div id="large-menu" key="large">
        <button class="bar-button" @click="goTraining">
          <tt>
            <i class="fa fa-home" aria-hidden="true"></i>
            Dash Board
          </tt>
        </button>
        <button class="bar-button" @click="goPrediction">
          <tt>
            <i class="fa fa-area-chart" aria-hidden="true"></i>
            Prediction
          </tt>
        </button>
      </div>
    </div>
  </transition>
</div>
</template>

<script>

export default {
  name: "NavigationBar",
  methods: {
    goTraining: function() {
      this.$store.commit("setPageName", {"page_name": "Training"});
      this.$router.push({path: "/"});
      this.hideMenu();
    },
    goPrediction: function() {
      this.$store.commit("setPageName", {"page_name": "Prediction"});
      this.$router.push({path: "/prediction"});
      this.hideMenu();
    },
    hideMenu: function() {
      this.$store.commit('setNavigationBarShowFlag', {flag: false});
    }
  }
}
</script>

<style lang="scss" scoped>
#navigation {
  $header-height: 44px;

  position: fixed;
  z-index: 999;

  #mask {
    position: fixed;
    top: $header-height;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 1;
    background-color: rgba(0, 0, 0, 0.4);
  }

  #navigation-bar {
    position: fixed;
    top: $header-height;
    height: calc(100vh - 35px);

    background-color: #262a4e;
    z-index: 2;

    #large-menu {
      padding-top: 35px;
      width: 180px;
    }

    .bar-button {
      text-align: left;
      width: 100%;
      height: 45px;
      margin: 2px 0px 0px 0px;
      background-color: #262a4e;
      color: #b7b7b7;
    }

    .bar-button:hover {
      color: #ffffff;
      background-color: #374b60;
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
