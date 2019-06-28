# RncInput

A Vue implementation of a ReNom Components input field.
This specifies single-line text entry field.


## Attributes
- value: bound item by v-model.
- [v-on]rnc-input: "action change" event. Changed input value, action event is raised.
- inputType: Specifies the type input element to display.
- onlyNumber: When this is true, the component only accept positive numbers including decimal.
- onlyInt: When this is true, the component only accept positive integers.
- checked: Checked state can be specified when inputType is "checkbox".
- invalidInput: When this is true, the component would be "worn" style. You can add conditions which to be error.
- disabled: When set to true, the input field should be unusable.
- placeHolder: The expected value of an input field.
- inputMaxLength: The maximum number of character the input string.
- inputMinLength: The minimum number of character the input string.
- inputMaxValue: The maximum value of the input value.
- inputMinValue: The maximum value of the input value.

## UNIT TEST

Render:
- Component it renders

Props default:
- Expected existence of 『invalidInput』
- props, for『invalidInput』expecting default as: 『false』

- Expected existence of 『disabled』
- props, for『disabled』expecting default as: 『false』

- Expected existence of 『placeHolder』
- props, for『placeHolder』expecting default as: 『""』

- Expected existence of 『inputMaxLength』
- props, for『inputMaxLength』expecting default as: 『0』

- Setting one prop: expecting existence of 『invalidInput』
- Setting props: expecting『invalidInput』to be: 『true』

- Setting one prop: expecting existence of 『disabled』
- Setting props: expecting『disabled』to be: 『true』

- Setting one prop: expecting existence of 『placeHolder』
- Setting props: expecting『placeHolder』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『inputMaxLength』
- Setting props: expecting『inputMaxLength』to be: 『66』

- Test of emitter by input element:『.input-in』. Expecting to emit『RncInput』 with the data: 『{"errorMessage":"","maxValueError":false,"minValueError":false,"textIncludedError":false,"value":""}』
