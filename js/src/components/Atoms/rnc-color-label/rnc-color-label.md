# RncColorLabel

A Vue implementation of a ReNom Components.
A set of colored box and text.


## Attributes
- title : Any string.
- color-class : Style class name. Defied in unified.scss. Among 'color-train', 'color-valid', 'color-created', 'color-reserved', 'color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5'.
- line-through : Boolean value. When 'true', text-decoration would be added.
- is-toggle-label : Boolean value. When 'true', cursor pointer would be added.
- @click="ToggleLabel()" : listen to clickEvent and do something.

## UNIT TEST of: RncColorLabel

Render:
- Component it renders

Props default:
- Expected existence of 『title』
- props, for『title』expecting default as: 『"Undefined"』

- Expected existence of 『colorClass』
- props, for『colorClass』expecting default as: 『"color-0"』

- Expected existence of 『lineThrough』
- props, for『lineThrough』expecting default as: 『false』

- Expected existence of 『isToggleLabel』
- props, for『isToggleLabel』expecting default as: 『false』

Set list of Props:
- Setting one prop: expecting existence of 『title』
- Setting props: expecting『title』to be: 『"Test text"』

- Setting one prop: expecting existence of 『colorClass』
- Setting props: expecting『colorClass』to be: 『"color-created"』

- Setting one prop: expecting existence of 『lineThrough』
- Setting props: expecting『lineThrough』to be: 『true』

- Setting one prop: expecting existence of 『isToggleLabel』
- Setting props: expecting『isToggleLabel』to be: 『true』
