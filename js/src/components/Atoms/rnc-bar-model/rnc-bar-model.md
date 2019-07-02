# RncBarModel

A Vue implementation of a ReNom Bar which shows the ratio of created models.



## Attributes

- modelInfo: The array of array which contain information of models.
　This is usually created by the getter "getFilteredModelList" and the method "reduceModelList".  


## UNIT TEST of: RncBarModel

Render:
- Component it renders

Props default:
- Expected existence of 『modelInfo』
- props, for『modelInfo』expecting default as: 『undefined』

- Expected existence of 『barWrapWidth』
- props, for『barWrapWidth』expecting default as: 『550』

- Expected existence of 『barWrapHeight』
- props, for『barWrapHeight』expecting default as: 『20』

Set list of Props:
- Setting one prop: expecting existence of 『modelInfo』
- Setting props: expecting『modelInfo』to be: 『666』

- Setting one prop: expecting existence of 『barWrapWidth』
- Setting props: expecting『barWrapWidth』to be: 『66』

- Setting one prop: expecting existence of 『barWrapHeight』
- Setting props: expecting『barWrapHeight』to be: 『66』

- Test of method. Expected existence of 『getColor』method
- Expected method 『getColor』to be a Function
- Expected method 『getColor』 with arguments 『66』to return 『"color-6"』

- Test of method. Expected existence of 『getWidth』method
- Expected method 『getWidth』to be a Function
- Expected method 『getWidth』 with arguments 『66』to return 『"width : 6600%;"』
