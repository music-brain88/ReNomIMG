<template>
  <div id="tag-item">
    <div id="tag-id" :style="{backgroundColor: tagColor(this.tagId)}">
      {{ tagId }}
    </div>
    <div id="tag-title">
      {{ tagName }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import { getTagColor } from '@/utils.js'

export default {
  name: 'TagItem',
  props: {
    tagName: '',
    tagId: '',
  },
  methods: {
    tagColor: function (n) {
      return getTagColor(n)
    }
  }
}
</script>

<style lang='scss'>
#tag-item {
  display: flex;
  align-items: center;
  width: calc(100% - 10px);
  height: 6%;
  margin-left: 5px;
  margin-right: 5px;
  border-bottom: solid 1px lightgray;
  #tag-id {
    width: 15%;
    color: white;
    text-align: center;
    margin-right: 10px;
  }
  #tag-title {
    color: gray;
  }
}
</style>
