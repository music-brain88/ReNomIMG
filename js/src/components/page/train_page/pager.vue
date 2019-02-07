<template>
  <div
    id="pager"
    tabindex="0"
    @keyup.right="nextPage"
    @keyup.left="prevPage">

    <!--Left Arrow-->
    <div
      class="pager-arrow"
      @click="prevPage">
      <i
        class="fa fa-caret-left"
        aria-hidden="true"/>
    </div>

    <!--Number-->
    <div
      v-for="(item, key) in pageList()"
      :class="{number: item !== '...'}"
      :style="pagerStyle(item)"
      :key="key"
      class="pager-number"
      @click="setPageNum(item)">
      {{ item }}
    </div>

    <!--Right Arrow-->
    <div
      class="pager-arrow"
      @click="nextPage">
      <i
        class="fa fa-caret-right"
        aria-hidden="true"/>
    </div>
  </div>
</template>

<script>

export default {
  name: 'Pager',
  props: {
    pageMax: {
      type: Number,
      default: 0
    },
    onSetPage: {
      type: Function,
      default: undefined,
    },
  },
  data: function () {
    return {
      pageIndex: 0
    }
  },
  computed: {
  },
  methods: {
    pagerStyle: function (index) {
      const current_page = this.pageIndex
      if (current_page === index) {
        return {
          'background-color': '#063662',
          'color': 'white',
        }
      }
    },
    setPageNum: function (index) {
      /**
        If the pushed pager button is number,
        set current page number as new number.
      */
      if (index === '...') return
      const max_page_num = this.pageMax - 1
      const current_page = this.pageIndex
      if (index === current_page) return
      this.pageIndex = Math.max((Math.min(index, max_page_num)), 0)
      if (this.onSetPage !== undefined) {
        this.onSetPage(this.pageIndex)
      }
    },
    nextPage: function () {
      /**
        Go to next page.
      */
      const index = this.pageIndex
      const max_page_num = this.pageMax - 1
      this.pageIndex = Math.max((Math.min(index + 1, max_page_num)), 0)
      if (this.onSetPage !== undefined) {
        this.onSetPage(this.pageIndex)
      }
    },
    prevPage: function () {
      /**
        Go to previous page.
      */
      const index = this.pageIndex
      const max_page_num = this.pageMax - 1
      this.pageIndex = Math.max((Math.min(index - 1, max_page_num)), 0)
      if (this.onSetPage !== undefined) {
        this.onSetPage(this.pageIndex)
      }
    },
    pageList: function () {
      /**
        Get the pager list.
      */
      const current_page = Math.max(this.pageIndex, 0)
      const max_page_num = Math.max(this.pageMax - 1, 0)

      if (max_page_num > 5) {
        if (current_page < 5) {
          return [...[...Array(Math.max(current_page, 7)).keys()], '...', max_page_num]
        } else if (current_page > max_page_num - 5) {
          return [0, '...', ...[...Array(Math.max(max_page_num - current_page, 7)).keys()].reverse().map(i => max_page_num - i)]
        } else {
          return [0, '...', ...[...Array(5).keys()].reverse().map(i => current_page - i + 2), '...', max_page_num]
        }
      } else {
        return Array(max_page_num).keys()
      }
    },
  }
}
</script>

<style lang='scss'>
#pager {
  width: 100%;
  height: 5%;
  display: flex;
  justify-content: flex-end;
  padding-right: 5px;
  align-items: center;
  .pager-arrow {
    font-size: 150%;
    height: 100%;
    display: flex;
    align-items: center;
    margin-left: 5px;
    margin-right: 5px;
    cursor: pointer;
    color: lightgray;
    transition: all 0.02s;
    i {
    }
    &:hover {
      color: gray;
    }
  }
  .pager-number {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 75%;
    height: calc(100% - 3px);
    margin-top: 2px;
    width: 3%;
    letter-spacing: -1px;
    color: gray;
  }
  .number {
    transition: all 0.1s;
    cursor: pointer;
    &:hover {
      color: black;
    }
  }
}
</style>
