# RncPager

A Vue implementation of a ReNom Components.

※ When component is active, an event occurs even if you press "←" "→" on the keyboard.  
※ Basically, the width is responsive, but "max-width: 30 px;" is specified for one element.  
※ The former prop "onSetPage" emits "SetPageFunction" and changed to the method to be executed on the parent side.  
※ As "justify-content: flex-end;" which was in "#pager {}" of the old style was deleted, please specify the placement on the parent side.  


## Attributes
- page-max: Please specify the number of pages. (Display starts from 0)
- SetPage($event): Page transition function.

## UNIT TEST of: RncPager

Render:
- Component it renders

- Setting one prop: expecting existence of 『pageMax』
- Setting props: expecting『pageMax』to be: 『66』

- Test of emitter by key event 『keyup.right』. Expecting to emit『set-page』 with the data: 『[1]』

- Test of emitter by key event 『keyup.left』. Expecting to emit『set-page』 with the data: 『[0]』

- Test of emitter by method on 『setPageNum』. Expecting to emit『set-page』 with the data: 『[1]』

- Expecting to emit『set-page』 with the data: 『[1]』
when method 『setPageNum』 is executed wutrh the arguments 『[1]』.

- Test of emitter by method on 『nextPage』. Expecting to emit『set-page』 with the data: 『[1]』

- Expecting to emit『set-page』 with the data: 『[1]』
when method 『nextPage』 is executed wutrh the arguments 『[]』.

- Test of emitter by method on 『prevPage』. Expecting to emit『set-page』 with the data: 『[0]』

- Expecting to emit『set-page』 with the data: 『[0]』
when method 『prevPage』 is executed wutrh the arguments 『[]』.

- Test of method. Expected existence of 『pageList』method
- Expected method 『pageList』to be a Function
- Expected method 『pageList』 with arguments 『』to return 『{}』
