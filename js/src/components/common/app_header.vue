<template>
<header>
  <div class="global-menu-button" @click="toggleMenu">
    <i id="menu-toggle" class="fa fa-bars" aria-hidden="true"></i>
  </div>

  <div id="soft-symbol">
    <span class="renom">ReNom</span>
    <span class="current-page">IMG > {{ currentPage }}</span>
  </div>

  <!-- <div class="help-menu-button">
    <i class="fa fa-th" aria-hidden="true"></i>
  </div> -->
</header>
</template>

<script>
export default {
  name: 'AppHeader',
  computed: {
    currentPage () {
      return this.$store.state.page_name
    }
  },
  methods: {
    toggleMenu: function () {
      this.$store.commit('setNavigationBarShowFlag', {
        flag: !this.$store.state.navigation_bar_shown_flag
      })
    }
  }
}
</script>


<style lang="scss" scoped>
header {
  // $header-height: 44px;
  // $header-color: #1e264b;
  // $font-size: 15pt;
  // $font-color: #ffffff;
  //
  // $content-margin: 8px;

  display: flex;
  position: fixed;
  top: 0px;
  left: 0px;
  width: 100vw;
  height: $header-height;

  margin: 0;
  padding: 0;

  background-color: $header-color;
  z-index: 999;

  .global-menu-button {
    margin-left: calc(#{$content-margin}*2);
    line-height: $header-height;
    vertical-align: middle;

    #menu-toggle:hover {
      color: #bbbbbb;
    }
  }

  .help-menu-button {
    margin-left: auto;
    margin-right: calc(#{$content-margin}*4);
    line-height: $header-height;
    vertical-align: middle;
  }

  #soft-symbol {
    margin: 0;
    text-align: left;
    color: $font-color;

    .renom, .current-page {
      line-height: $header-height;
      font-size: $font-size;
      vertical-align: middle;
    }
    .renom {
      margin-left: $content-margin;
      font-weight: bold;
    }
  }
}
</style>
