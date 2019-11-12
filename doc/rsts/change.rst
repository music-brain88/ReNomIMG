Change Log
===========


**v2.3.2**

- Fixed bug where model ID information was displayed as 'undefined' when stopping model training on GUI
- Fixed bug in GUI Prediction Result display where 'Image' toggle status on Segmentation Page affected whether or not images were displayed on the Classification and Detection pages
- Modified error message displayed on GUI when training diverges and model parameters cause a numerical overflow to be more descriptive and suggest alternative actions to take
- Fixed incorrect batch normalization layer momentum value for Yolo v1, Yolo v2, ResNet and ResNeXt models
- Fixed bug in Yolo v2 Python API loss function that caused error when image height and width were not equal to each other

**v2.3b2**

- Fixed bug in Grad-CAM API that caused error when calculating forward propagation in model
- Fixed bug in forward propagation method definition that allowed model to calculate output even when number of classes were undefined

**v2.3b1**

- Added prediction scores column to prediction.csv output on "Predict" GUI page for Classification models
- Changed Model Distribution plot to only show blinking point for model that is currently training
- Changed Model Distribution plot to remove points for model in CREATED or RESERVED states
- Fixed bug in classmap loading function for SSD object detection algorithm
- Fixed bug in Deeplabv3+ segmentation algorithm classmap definition
- Fixed bug in model state where model would not enter RESERVED state

**v2.3b0**

- Added Deeplabv3+ segmentation model to GUI
- Revised GUI component details

**v2.2b1**

- Made GUI text selectable
- Fixed bugs

**v2.2b0**

- Added Deeplabv3+ segmentation model to API
- Refactored GUI components
- Refactored backend server
- Refactored CNN model architecture code
- Modified Yolov2 loss function
- Modified Yolov1 and Yolov2 pretrained weights
- Fixed bugs

**v2.1b3**

- Fixed bugs
- Revised documentation

**v2.1b2**

- Added Grad-CAM visualization tool
- Added pytests
- Revised README.md format
- Added URLs to download past wheel packages to README.md
- Fixed bugs

**v2.1b1**

- Modify img loader to accept binary image

**v2.1b0**

- Add new augmentation methods
- Add a function for downloading the prediction result as csv
- Modify image data preprocess pipeline
- Update Node.js and Python dependencies
- Fix UI Bugs

**v2.0.4**

- Update dependencies

**v2.0.3**

- Fix UI Bugs

**v2.0.2**

- Add warning to dataset create modal(GUI) when illegal train valid ratio is inputted
- Add error handler to renom_img.api.inference.detector.Detector

**v2.0.1**

- Fixed UI bugs
- Updated webpack@3.x => webpack@4.x
- Modified eslint settings
- Modified Darknet architecture (Added BatchNormalization and removed bias term from convolution layer)
- Modified Yolov1 loss function
