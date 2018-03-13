<template>
  <div id="tag-list">
    <div class="title">
      Tags
    </div>

    <div class="content">
      <tag-item v-for="(name, id) in labels" :key="id" :cls_id="id" :name="name">
      </tag-item>
    </div>
  </div>
</template>

<script>
import TagItem from './tag_list_parts/tag_item.vue'

export default {
  name: "TagList",
  components: {
    "tag-item": TagItem
  },
  computed: {
    labels() {
      return this.$store.getters.getLabelDict
    }
  }
}
</script>

<style lang="scss" scoped>
#tag-list {
  $component-margin-top: 32px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-padding-top: 36px;
  $content-padding-horizontal: 32px;
  $content-padding-bottom: 36px;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .content {
    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;
  }
}
</style>

