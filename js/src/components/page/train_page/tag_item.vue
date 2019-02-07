<template>
  <div id="tag-item">
    <div
      id="tag-id"
      :style="{backgroundColor: tagColor(tagId)}">
      {{ tagId }}
    </div>
    <div id="tag-title">
      {{ tagName }}
    </div>
  </div>
</template>

<script>
import { getTagColor } from '@/utils.js'

export default {
  name: 'TagItem',
  props: {
    tagName: {
      type: String,
      default: ''
    },
    tagId: {
      type: Number,
      default: 0
    },
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
    height: 60%;
    color: white;
    text-align: center;
    margin-right: 10px;
    font-size: 0.7rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #tag-title {
    color: gray;
    font-size: 0.8rem;
  }
}
</style>
