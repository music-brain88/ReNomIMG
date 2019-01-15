Introduction
============

**ReNomIMG is a GUI tool & PythonAPI for building image recognition models.**

.. image:: /_static/image/top.png

1. Concept
----------

.. ユーザが自分自身で目的に沿ったAIモデルを作れるようにすること.

The concept of ReNomIMG is **ensure that users can create AI models 
according to the purpose by themselves**.

Recent developing deep learning technology realizes extremely big improvement at
recognition accuracy.  

However if you would create a recognition model for any business scene such as 
recognising damages of manufactured products, there are still many problems for 
earning high accuracy recognition model.

For example, correcting training dataset, programming recognition model and train it, 
evaluating the model, and so on.

Especially, even if deep learning era, it is required to tune up the hyper parameters of 
the recognition model. It requires many try and errors.

ReNomIMG allows you to build object detection model easily.

2. What ReNomIMG provides you.
-------------------------------

ReNomIMG provides gui tool and python api.

GUI tool
~~~~~~~~~~~~~~

ReNomIMG GUI tool allows you to build object detection models.
What users have to do are **preparing training data**, 
**setting train configuration** and **pushing run button**.


.. 下の図は, 後で差し替え

.. image:: /_static/image/renomimg_gui_top.png

3. ReNomIMG overview
---------------------

ReNomIMG is service for image classification.Image classification has three types

* Object Detection(or just detection).
* Semantic Segmentation(or just Segmentation).
* Image Classification(or just classification).

ReNomIMG can do these three kind of tasks. Not only one job

ReNomIMG is not only that, such as detection, segmentation, classification.
You can also to use own data and splitting, model comparing, to see how to learning data and
export model then use another services.

4. ReNomIMG Algorithms
----------------------

You can use these algorithms.

* detection

  - Yolo v1
  - Yolo v2
  - SSD

* segmentation

  - U-Net
  - FCN
  - TernousNet

* classification

  - ResNet
  - ResNeXt
  - VGG


5. Python API
---------------------
ReNomIMG API is a python api which provides you not only modern **object detection model** 
but also **classification model**, **segmentation model**. 

And more, all those models have pretrained weights.
This makes models more accurate one.

An example code is bellow. Using ReNomIMG, you can build a model and train it in 3 lines.

**Building a VGG16 Model**

.. code-block :: python
    :linenos:
    :emphasize-lines: 12,13,16

    from renom_img.api.classification.vgg import VGG16
    from renom_img.api.utility.load import parse_xml_detection
    from renom_img.api.utility.misc.display import draw_box

    ## Data preparation.
    train_image_path_list = ...
    train_label_list = ...
    valid_image_path_list = ...
    valid_label_list = ...

    ## Build a classification model(ex: VGG16).
    model = VGG16(class_map, load_pretrained_weight=True, train_whole_network=False)
    model.fit(train_image_path_list, train_label_list, valid_image_path_list, valid_label_list)

    ## Prediction.
    prediction = model.predict(new_image)

