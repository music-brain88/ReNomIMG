# RncTagListItem

A Vue implementation of a ReNom Components.

It is a specification that fits the width of the user.



## Attributes

- tag-name: If the name is too long, the design may be broken.
- tag-id: The background color is determined by the ones place of the input value.
- item-height: It is possible to change the height.

## UNIT TEST of: RncTagListItem

Render:
- Component it renders

Props default:
- Expected existence of 『tagName』
- props, for『tagName』expecting default as: 『"Undefined"』

- Expected existence of 『tagId』
- props, for『tagId』expecting default as: 『0』

- Expected existence of 『itemHeight』
- props, for『itemHeight』expecting default as: 『28』

Set list of Props:
- Setting one prop: expecting existence of 『tagName』
- Setting props: expecting『tagName』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『tagId』
- Setting props: expecting『tagId』to be: 『66』

- Setting one prop: expecting existence of 『itemHeight』
- Setting props: expecting『itemHeight』to be: 『666』

- Test of computed property. Test existence of 『tagColor』
Setting props 『{"tagId":0}』, expecting 『"#E7009A"』
