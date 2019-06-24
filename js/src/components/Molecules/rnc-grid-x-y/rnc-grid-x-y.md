# RncGridXY

A Vue implementation of a ReNom Components.

※ Horizontal width and vertical width fit the size specified on the parent side.  
※ Scaling is possible by scrolling operation.  

#### 【LearningCurve】
※ "SelectedModelObj": Item acquired by "getSelectedModel" (Getter) in ReNomIMG.   
　(Here, only the necessary items are extracted)  
※ The display position of "best_epoch_valid_result.nth_epoch" is shifted by 1,  
　but due to the original specifications of ReNomIMG, leave it as it is.  

#### 【ModelScatter】
※ "FilteredAndGroupedModelListArray" of attr is  
　acquired by "getFilteredAndGroupedModelList" of ReNomIMG.  
※ As it is included in the above getter,  
　it is not necessary to specify "AxisName".  




## Attributes

#### 【LearningCurve】
- axis-name-x: X axis name  
- axis-name-y: Y axis name  
- switch-train-graph: You can set display / non-display of train graph.
- switch-valid-graph: You can set display / non-display of valid graph.
- selected-model-obj: This item is acquired by "getSelectedModel" (Getter) in ReNomIMG. (Here, only necessary items are extracted)  

#### 【ModelScatter】
- FilteredAndGroupedModelListArray: "getFilteredAndGroupedModelList" (Getter) of ReNomIMG. (Here, only necessary items are extracted)  


## UNIT TEST of: RncGridXY

Render:
- Component it renders

Props default:
- Expected existence of 『AxisNameX』
- props, for『AxisNameX』expecting default as: 『""』

- Expected existence of 『AxisNameY』
- props, for『AxisNameY』expecting default as: 『""』

- Expected existence of 『SelectedModelObj』
- props, for『SelectedModelObj』expecting default as: 『undefined』

- Setting one prop: expecting existence of 『AxisNameX』
- Setting props: expecting『AxisNameX』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『AxisNameY』
- Setting props: expecting『AxisNameY』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『SelectedModelObj』
- Setting props: expecting『SelectedModelObj』to be: 『{}』
