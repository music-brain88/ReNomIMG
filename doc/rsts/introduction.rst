Introduction
============

**ReNomIMG is a GUI tool & Python API for building image recognition models.**

.. image:: /_static/image/top.png

1. Concept
----------

.. ユーザが自分自身で目的に沿ったAIモデルを作れるようにすること.

The concept of ReNomIMG is to **allow users to easily create AI models
tailored to their objectives**.

Rapid developments in deep learning have resulted in tremendous improvements
in image recognition accuracy.

However, creating accurate image recognition models for specific business use, such as
damage detection in manufactured products, is difficult due to a number of obstacles.

These obstacles include gathering a sufficiently large, accurate training dataset,
programming the model, training the model, evaluating its performance, and so on.

Additionally, even with recent deep learning models, proper tuning of the
the model hyper-parameters is very important. This usually requires significant trial and error.

ReNomIMG allows you to build image recognition models easily without these hassles.

2. ReNomIMG overview
---------------------

ReNomIMG is an application for building and training deep-learning image recognition models.
Image recognition is commonly divided into three tasks:

* Object Detection (or 'detection')
* Semantic Segmentation (or 'segmentation')
* Image Classification (or 'classification')

ReNomIMG allows users to perform all three of these tasks.

ReNomIMG also provides users dataset creation and partition features, model evaluation metrics,
training curve visualizations and the ability to export models for service integration.

3. What ReNomIMG provides you
-------------------------------

ReNomIMG provides you with a user-friendly GUI tool, a variety of popular algorithms, and a flexible python API.

GUI tool
~~~~~~~~~~~~~~

The ReNomIMG GUI tool allows users to build object detection, semantic segmentation, and image classification models.
Users can do this simply by **preparing training data**,
**selecting the training configuration** and **pushing the run button**.


.. 下の図は, 後で差し替え

.. image:: /_static/image/renomimg_gui_top.png

ReNomIMG Algorithms
~~~~~~~~~~~~~~

ReNomIMG provides the following algorithms.

* Detection

  - Yolo v1
  - Yolo v2
  - SSD

* Segmentation

  .. - U-Net
  - FCN
  - Deeplabv3
  .. - TernousNet

* Classification

  - ResNet
  - ResNeXt
  - VGG


Python API
~~~~~~~~~~~~~~

ReNomIMG includes a flexible python API that provides a simple interface for popular **object detection**,
**semantic segmentation**, and **image classification models**.

Pre-trained weights are also available, allowing users to efficiently train high-performing models.

An example of defining and training a model is shown below. With ReNomIMG you can build a model, train it, and make predictions in just 3 lines.

**Building a VGG16 Model**

.. code-block :: python
    :linenos:
    :emphasize-lines: 13,14,17

    from renom_img.api.classification.vgg import VGG16

    ## Data preparation
    train_image_path_list = ...
    train_label_list = ...
    valid_image_path_list = ...
    valid_label_list = ...

    ## Define your dataset classes
    class_map = ...

    ## Build a classification model(ex: VGG16)
    model = VGG16(class_map, load_pretrained_weight=True, train_whole_network=False)
    model.fit(train_image_path_list, train_label_list, valid_image_path_list, valid_label_list)

    ## Prediction
    prediction = model.predict(new_image)

