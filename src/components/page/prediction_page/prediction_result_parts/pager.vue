<template>
  <div v-if="getPredictResults.length > 0" class="pager">
    <div v-if="(currentPage - 1) >= 0" class="page_nav_button" @click="setPredictPage(0)"><<</div>
    <div v-if="(currentPage - 1) >= 0" class="page_nav_button" @click="setPredictPage(currentPage - 1)"><</div>
    <div v-if="(currentPage - 2) >= 0" class="page_nav_button" @click="setPredictPage(currentPage - 2)">{{ currentPage - 2 }}</div>
    <div v-if="(currentPage - 1) >= 0" class="page_nav_button" @click="setPredictPage(currentPage - 1)">{{ currentPage - 1 }}</div>
    <div @click="setPredictPage(currentPage)" class="page_nav_button current">{{ currentPage }}</div>
    <div v-if="(currentPage + 1) <= pageMax" class="page_nav_button" @click="setPredictPage(currentPage + 1)">{{ currentPage + 1 }}</div>
    <div v-if="(currentPage + 2) <= pageMax" class="page_nav_button" @click="setPredictPage(currentPage + 2)">{{ currentPage + 2 }}</div>
    <div v-if="(currentPage + 1) <= pageMax" class="page_nav_button" @click="setPredictPage(currentPage + 1)">></div>
    <div v-if="(currentPage + 1) <= pageMax" class="page_nav_button" @click="setPredictPage(pageMax)">>></div>
  </div>
</template>

<script>
export default {
  name: "Pager",
  computed: {
    getPredictResults: function () {
      return this.$store.getters.getPredictResults;
    },
    currentPage: function() {
      return this.$store.state.predict_page;
    },
    pageMax: function() {
      return this.$store.getters.getPageMax;
    }
  },
  methods: {
    setPredictPage(page) {
      this.$store.commit("setPredictPage", {
        "page": page,
      });
    }
  }
}
</script>

<style lang="scss" scoped>
.pager {
  $nav_button_width: 20px;
  $nav_button_height: 20px;

  $nav_button_font_size: 12px;
  $nav_button_color: #999999;

  $nav_button_bg_color: #ffffff;

  $nav_button_border_color: #cccccc;
  $nav_button_border_width: 1px;
  $nav_button_border_radius: 2px;

  display: flex;
  margin-left: auto;
  margin-top: 16px;

  .page_nav_button {
    width: $nav_button_width;
    height: $nav_button_height;
    margin-left: 4px;
    line-height: $nav_button_height;
    font-size: $nav_button_font_size;
    font-weight: 500;
    color: $nav_button_color;
    text-align: center;
    background-color: $nav_button_bg_color;
    border: $nav_button_border_width solid $nav_button_border_color;
    border-radius: $nav_button_border_radius;
    cursor: pointer;
  }
  .current {
    background-color: $nav_button_color;
    color: $nav_button_bg_color;
  }
}
</style>