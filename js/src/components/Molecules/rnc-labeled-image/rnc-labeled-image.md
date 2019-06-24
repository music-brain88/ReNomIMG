# RncLabeledImage

A Vue implementation of a ReNom Component.
It renders the set of image and labels of annotation.



## Attributes
- width: An original width value of image. Use for the style calculation.
- height: An original height value of image. Use for the style calculation.
- maxWidth: A restriction of the image width.
- maxHeight: A restriction of the image height.
- img: An image source.
- model: A "model" instance.  
- dataset: A "dataset" instance.
- result: An object of annotation information.
- callback: callback used in onImageClick() and drawSeg().
- boxEnterCallback: Not using currently.
- boxLeaveCallback: Not using currently.
- showPredict: A boolean value for showing prediction results.
- showTarget: A boolean value for showing the supervised labels.
- showImage: A boolean value for showing the image. Use in modal_image.vue.

## UNIT TEST of: RncLabeledImage

Render:
- Component it renders

Props default:
- Expected existence of 『width』
- props, for『width』expecting default as: 『0』

- Expected existence of 『height』
- props, for『height』expecting default as: 『0』

- Expected existence of 『maxWidth』
- props, for『maxWidth』expecting default as: 『0』

- Expected existence of 『maxHeight』
- props, for『maxHeight』expecting default as: 『0』

- Expected existence of 『img』
- props, for『img』expecting default as: 『""』

- Expected existence of 『model』
- props, for『model』expecting default as: 『null』

- Expected existence of 『dataset』
- props, for『dataset』expecting default as: 『null』

- Expected existence of 『result』
- props, for『result』expecting default as: 『{"index":-1}』

- Expected existence of 『callback』
- props, for『callback』expecting default as: 『null』

- Expected existence of 『boxEnterCallback』
- props, for『boxEnterCallback』expecting default as: 『null』

- Expected existence of 『boxLeaveCallback』
- props, for『boxLeaveCallback』expecting default as: 『null』

- Expected existence of 『showPredict』
- props, for『showPredict』expecting default as: 『false』

- Expected existence of 『showTarget』
- props, for『showTarget』expecting default as: 『true』

- Expected existence of 『showImage』
- props, for『showImage』expecting default as: 『true』

- Setting one prop: expecting existence of 『width』
- Setting props: expecting『width』to be: 『66』

- Setting one prop: expecting existence of 『height』
- Setting props: expecting『height』to be: 『66』

- Setting one prop: expecting existence of 『maxWidth』
- Setting props: expecting『maxWidth』to be: 『666』

- Setting one prop: expecting existence of 『maxHeight』
- Setting props: expecting『maxHeight』to be: 『666』

- Setting one prop: expecting existence of 『img』
- Setting props: expecting『img』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『model』
- Setting props: expecting『model』to be: 『{"test":"TEST TEXT"}』

- Setting one prop: expecting existence of 『dataset』
- Setting props: expecting『dataset』to be: 『{"test":"TEST TEXT"}』

- Setting one prop: expecting existence of 『result』
- Setting props: expecting『result』to be: 『{"index":0,"target":[{"box":[0.55859375,0.35411764705882354,0.5609375,0.6235294117647059],"name":"cat","size":[640,425],"class":0}],"predict":[{"class":0,"name":"cat","box":[0.55859375,0.35411764705882354,0.1609375,0.2235294117647059],"score":0.3106231093406677}]}』

- Setting one prop: expecting existence of 『callback』
- Setting props: expecting『callback』to be: 『undefined』

- Setting one prop: expecting existence of 『boxEnterCallback』
- Setting props: expecting『boxEnterCallback』to be: 『undefined』

- Setting one prop: expecting existence of 『boxLeaveCallback』
- Setting props: expecting『boxLeaveCallback』to be: 『undefined』

- Setting one prop: expecting existence of 『showPredict』
- Setting props: expecting『showPredict』to be: 『true』

- Setting one prop: expecting existence of 『showTarget』
- Setting props: expecting『showTarget』to be: 『false』

- Setting one prop: expecting existence of 『showImage』
- Setting props: expecting『showImage』to be: 『false』
