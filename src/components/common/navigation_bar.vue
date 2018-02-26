<template>
  <div id="navigation-bar">
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
      this.$store.commit('setNavigationBarShowFlag', {
        flag: false,
      })
    }
  }
}
</script>

<style lang="scss">

  #navigation-bar {
    background-color: #262a4e;
    height: calc(100vh - 35px);

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

</style>
