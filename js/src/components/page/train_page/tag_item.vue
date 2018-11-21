<template>
  <div id="tag-item">
    <div id="tag-id" :style="{backgroundColor: getTagColor(this.tagId)}">
      {{ tagId }}
    </div>
    <div id="tag-title">
      {{ tagName }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'TagItem',
  props: {
    tagName: '',
    tagId: '',
  },
  components: {
  },
  computed: {
    ...mapGetters(['getTagColor']),
  },
  created: function () {

  },
  methods: {

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
