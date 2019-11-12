How to use the ReNomIMG GUI
============================

Start the Application
----------------------

ReNomIMG is a single page web application.
After installation is finished, you can run the application
in any directory with the following commands.

.. code-block :: shell

    cd workspace # The workspace can be any directory in your PC.
    renom_img # This command starts the ReNomIMG GUI server.

The command ``renom_img`` accepts the following arguments.

* ``--host`` : This specifies the server address.
* ``--port`` : This specifies the server port number.

For example, the following command runs ReNomIMG on port 8888.

.. code-block :: shell

    renom_img --port 8888 # Run ReNomIMG on port 8888

After starting the application server, open your web browser and type the
server address into the address bar.

.. image:: /_static/image/how_to_use_start.png

The ReNomIMG GUI will load in your web browser.


.. _dir_structure:

Provide Image and Label Data
------------------

When the server starts, it will create ``datasrc`` and ``storage`` directories
in the current running directory if they don't already exist.

The ``datasrc`` directory has the following folder structure.

.. code-block :: shell

    datasrc/
      ├── img   # Place training img files here.
      ├── label
      │   ├── classification # Place classification training label files here
      │   ├── detection      # Place detection training label files here
      │   └── segmentation   # Place segmentation training label files here
      └── prediction_set
          └── img   # Place prediction img files here.

As shown in the structure above, please place the training image data in ``datasrc/img``,
and the training label data in ``datasrc/label``.

.. note::

    The name of the image file and corresponding label file name must be the same.
    For example, for object detection data if an image file name is ``image01.jpg``,
    the corresponding label file name must be ``image01.xml``.


Format of Detection data
~~~~~~~~~~~~~~~~~~~

Object detection image and label files should conform to the formats below.

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

**Format of label files** : ReNomIMG only accepts ``xml`` formatted label files.
The format of the xml file is shown below.

**Place xml files here**: ``<ReNomIMG dir>/datasrc/label/detection/<sample.xml>``

.. code-block :: shell

    <annotation>
    	<size>
    		<width>374</width>
    		<height>500</height>
    		<depth>3</depth>
    	</size>
    	<object>
    		<name>car</name>
    		<bndbox>
    			<xmin>2</xmin>
    			<ymin>3</ymin>
    			<xmax>374</xmax>
    			<ymax>500</ymax>
    		</bndbox>
    	</object>
    </annotation>

ReNomIMG accepts the PASCAL VOC format for object detection data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
| 
|

Format of Classification data
~~~~~~~~~~~~~~~~~~~

Classification image and label files should conform to the formats below.

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

**Format of label files** : ReNomIMG only accepts ``txt`` formatted label files.
The format of the text file is shown below.

Please save the file as ``target.txt``.

**Place label file here**: ``<ReNomIMG dir>/datasrc/label/classification/target.txt``

.. code-block :: shell

    crayfish_image_0035.jpg crayfish
    crayfish_image_0065.jpg crayfish
    crayfish_image_0037.jpg crayfish
    crayfish_image_0032.jpg crayfish
    crayfish_image_0028.jpg crayfish
    crayfish_image_0051.jpg crayfish
    wrench_image_0035.jpg wrench
    wrench_image_0037.jpg wrench
    wrench_image_0032.jpg wrench
    wrench_image_0028.jpg wrench
    wrench_image_0019.jpg wrench
    wrench_image_0031.jpg wrench
    ...
    ...
    ...
    ...
    ...
    pigeon_image_0035.jpg pigeon
    pigeon_image_0037.jpg pigeon
    pigeon_image_0032.jpg pigeon
    pigeon_image_0028.jpg pigeon
    pigeon_image_0019.jpg pigeon
    pigeon_image_0031.jpg pigeon
    pigeon_image_0012.jpg pigeon
    pigeon_image_0002.jpg pigeon
    pigeon_image_0015.jpg pigeon
    pigeon_image_0042.jpg pigeon
    pigeon_image_0036.jpg pigeon
    pigeon_image_0022.jpg pigeon
    pigeon_image_0021.jpg pigeon
    pigeon_image_0029.jpg pigeon


ReNomIMG accepts the PASCAL VOC format for classification data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
|
|

Class numbers are assigned to each class based on the alphabetical order of the class names, beginning with 0.


Format of Segmentation data
~~~~~~~~~~~~~~~~~~~

Semantic segmentation image and label files should conform to the formats below.

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

.. warning::
    Segmentation requires two kinds of label data.
    ``PNG`` files (one per image) and ``class_map.txt`` (one per dataset).

**Format of label files** : ReNomIMG only accepts ``txt`` formatted files for
class labels and ``PNG`` files for image label data. The format of the txt file is shown below.

Please save the class label list as ``class_map.txt``.

**Place class label file here**: ``<ReNomIMG dir>/datasrc/label/segmentation/class_map.txt``

**Example of good data**

* Class id number starts at 0, which is set to the background class.
* Class id numbers are serially numbered.

.. code-block :: shell

       background 0
       airplane 1
       bicycle 2
       bird 3
       boat 4
       bottle 5
       bus 6
       car 7
       cat 8
       chair 8
       cow 10
       diningtable 11
       dog 12
       horse 13
       motorbike 14
       person 15
       potted plant 16
       sheep 17
       sofa 18
       train 19
       tv/monitor 20


----

.. raw:: html

  <style>.red {color:red} </style>

.. role:: red

:red:`Example of incorrect data`

* Class id number does not start at 0.
* Class names do not include a background class.
* Class numbers are not serially numbered.

.. code-block :: shell

       airplane 1
       bicycle 10
       bird 50
       boat 100
       bottle 150
       bus 200
       car 250
       cat 300
       chair 350
       cow 400
       diningtable 450
       dog 500
       horse 550
       motorbike 600
       person 700
       potted plant 750
       sheep 800
       sofa 900
       train 950
       tv/monitor 1000

The following is a sample segmentation label PNG file. The class id numbers
have been mapped to colors with a color map for visualization purposes.

.. image:: /_static/image/009592.png


ReNomIMG accepts the PASCAL VOC format for semantic segmentation data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
| 
|

.. note:: 
    The name of the image file and corresponding label file name must be the same.
    For example, if the image file name is ``image01.jpg``, the corresponding label file name
    must be ``image01.png``.


Create Model
----------------------

The application server and dataset are now both ready, so let's build an object detection model.
To build a model, you must specify the dataset and the training hyper-parameters.

Create Dataset
~~~~~~~~~~~~~~

To train a machine learning model, you should prepare training and validation sub-datasets.
The training sub-dataset is used for training the model, and the validation sub-dataset is used for
evaluating how accurately the model can predict data that has not been used in training.

In ReNomIMG, the training and validation sub-datasets will be **randomly** sampled from the data
that is in the ``datasrc`` directory.

.. image:: /_static/image/how_to_use_gui_datasrc.png

As shown in the figure above, you can create a dataset from the datasrc images.
Once a dataset is created its contents will never change.

For creating a dataset, please open the dataset modal. The following figures
guide you through this step.

.. image:: /_static/image/how_to_use_gui_dataset_create_button01.png

The following page is displayed next.

.. image:: /_static/image/how_to_use_gui_dataset_create_button02.png

As shown above, you can specify the dataset name, description and ratio of training to validation data.

After entering this information, click the ``Confirm`` button to generate the sub-datasets.

The following visual will be shown. You can confirm what classes exist in the dataset,
their ratios, and the total number of images.

.. image:: /_static/image/how_to_use_gui_dataset_create_button03.png


Finally, to save the dataset click the ``Submit`` button.

You can confirm all datasets you have created on the dataset page.
To access the dataset page, please follow the steps shown below.

.. image:: /_static/image/how_to_use_gui_dataset_create_button04.png

.. image:: /_static/image/how_to_use_gui_dataset_create_button05.png

In the figure above, 2 datasets have already been created.

Configure Hyper-parameters
~~~~~~~~~~~~~~~~~~~~~~~

After completing the steps above, you can build a model and start training it.
To create a model, click the button ``+New`` button.

.. image:: /_static/image/how_to_use_gui_model_create01.png

This will open a hyper-parameter configuration modal, as shown below.

.. image:: /_static/image/how_to_use_gui_model_create02.png

As seen in the figure, you can specify the following parameters.

* **Dataset Name** ... Select the dataset for training.
* **Algorithm** ... Select the CNN algorithm.
* **Batch Size** ... Set the batch size. A larger number can speed up training but requires more memory.
* **Total Epoch** ... Set the number of times your model should pass through the dataset during training. All images are seen once in every epoch.
* **Image Width** ... Image width for resizing images during training.
* **Image Height** ... Image height for resizing images during training.
* **Load pretrain weight** ... Check this box to load the pretrained weights for the algorithm as initial weight values. If unchecked, the weights are randomly initialized.
* **Train Whole network** ... Check this box to train all layers of the model. If unchecked, the pretrained layers will be frozen during training.
.. note::

    Depending on your GPU device, a larger image size or batch size may cause a memory overflow.


Train the Model
~~~~~~~~~~~~~~

After configuring the hyper-parameters, **click the Create button to start training!**

As training begins, the model will be added to the model list and the train progress bar will also appear.

.. image:: /_static/image/how_to_use_gui_model_create03.png

.. note::

  The same procedure for building and training a model can be used for Detection, Segmentation and Classification.


Perform Predictions
------------------

After training is finished, we can use the model for making predictions with new image data.

Click the ``Deploy`` button shown in the `Model Detail` window on the `Train Page`
to select which model to use for performing predictions on new data.
The currently deployed model is shown at the top of the model list with a status of 'Deployed'.

.. image:: /_static/image/how_to_use_gui_prediction_deploy_button.png

After deployment, open the `Predict` page using the side bar menu.
The following figure shows the `Predict` page.

.. image:: /_static/image/how_to_use_gui_prediction_button.png
    :scale: 80 %


To run the prediction using the deployed model, click the ``Run Prediction`` button.

.. note::

    The images used for predictions are all those contained in the `datasrc/prediction_set/img` directory.
    The directory structure is described in :ref:`Provide your dataset<dir_structure>`.


After the predictions are made, the results are displayed in the window.


.. image:: /_static/image/how_to_use_gui_prediction_result.png
    :scale: 80 %


You can also download the prediction results as a csv file. Click the ``Download`` button
on the top right to download the file.

.. image:: /_static/image/how_to_use_gui_prediction_download_button.png
    :scale: 70 %


Uninstall ReNomIMG
------------------

You can uninstall ReNomIMG with the following pip command.

.. code-block :: shell

    pip uninstall renom_img

~~~~~~~~~~~~~~~~~~~
