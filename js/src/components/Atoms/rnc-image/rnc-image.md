# RncImage

A Vue implementation of a ReNom Components.
It would render the image of which the source have been sent .


## Attributes

- img: The source of the image.
- imgWidth: The actual width of the image.
- imgHeight: The actual height of the image.
- maxWidth: The max width can be set manually and lately.
- maxHeight: The max height can be set manually and lately.
- showImage: Validate if the image exists.

## UNIT TEST of: RncImage

Render:
- Component it renders

Props default:
- Expected existence of 『img』
- props, for『img』expecting default as: 『""』

- Expected existence of 『imgWidth』
- props, for『imgWidth』expecting default as: 『0』

- Expected existence of 『imgHeight』
- props, for『imgHeight』expecting default as: 『0』

- Expected existence of 『maxWidth』
- props, for『maxWidth』expecting default as: 『0』

- Expected existence of 『maxHeight』
- props, for『maxHeight』expecting default as: 『0』

- Expected existence of 『showImage』
- props, for『showImage』expecting default as: 『true』

- Setting one prop: expecting existence of 『img』
- Setting props: expecting『img』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『imgWidth』
- Setting props: expecting『imgWidth』to be: 『66』

- Setting one prop: expecting existence of 『imgHeight』
- Setting props: expecting『imgHeight』to be: 『66』

- Setting one prop: expecting existence of 『maxWidth』
- Setting props: expecting『maxWidth』to be: 『666』

- Setting one prop: expecting existence of 『maxHeight』
- Setting props: expecting『maxHeight』to be: 『666』

- Setting one prop: expecting existence of 『showImage』
- Setting props: expecting『showImage』to be: 『true』

Test of error when set props 『{"imgWidth":"NOT A NUMBER","imgHeight":"NOT A NUMBER","maxWidth":"NOT A NUMBER","maxHeight":"NOT A NUMBER"}』

- Test of computed property. Test existence of 『modifiedSize』
Setting props 『{"imgWidth":66,"imgHeight":66,"maxWidth":666,"maxHeight":666}』, expecting 『{"width":"calc(666px - 0.4vmin)","height":"calc(666px - 0.4vmin)"}』
