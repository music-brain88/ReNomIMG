How to Use ReNomIMG GUI tool
============================

Start the Application
----------------------

ReNomIMG is a single page web application.
If your installation have done successfully, 
you can run application in any directory with following commands.

.. code-block :: shell

    cd workspace # Workspace can be any directory in your pc. 
    renom_img # This command will starts ReNomIMG GUI server.

For the command ``renom_img``, you can give following arguments.

* --host : This specifies server address.
* --port : This specifies port number of the server.

For example, following code runs ReNomIMG with port 8888.

.. code-block :: shell

    renom_img --port 8888 # Running ReNomIMG with port 8888

If the application server runs, open web browser and type the 
server address to the address bar like this.

.. image:: /_static/image/how_to_use_start.png

Then the application will be appeared.


Place your dataset
------------------

When the server starts, ``datasrc`` directory and ``storage`` directory
will be created in the server running directory.

The ``datasrc`` directory has following folder structure.

.. code-block :: shell

    datasrc/
      ├── img   # Set training img files here.
      ├── label
      │    ├── classofication # Set classofication training label files here.
      │    ├── detection      # Set detection training label files here.
      │    └── segmentation   # Set segmentation training label files here.
      └── prediction_set
            ├── img     # Set prediction img files here.
            └── output  # Prediction result will be output here.
            	  ├── classofication # Set classofication training label files here.
            	  ├── detection      # Set detection training label files here.
            	  │     ├── csv
                  │     └── xml
            	  └── segmentation   # Set segmentation training label files here.

As written in the above comments, please set training image data to ``datasrc/img``,
set training label data to ``datasrc/label``.

.. note::

    The name of image file and corresponded label file name have to be same.
    For example, the image file name is ``image01.jpg``, corresponded label file name
    have to be ``image01.xml``.


Format of the Detection data
~~~~~~~~~~~~~~~~~~~

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

**Format of label files** : ReNomIMG only accepts ``xml`` formatted label files.
The format of xml file is bellow.

**put xml files here**:``<ReNomIMG dir>datasrc/label/detection/<sample.xml>``

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

ReNomIMG accepts PASCAL VOC formatted object detection data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
| 
|

Format of the Classification data
~~~~~~~~~~~~~~~~~~~

**Format of label files** : ReNomIMG only accepts ''txt'' formatted label files.
The format of text file is bellow.

Please Save as target.txt

**put here**:``<ReNomIMG dir>datasrc/label/classification/target.txt``

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
 
ReNomIMG accepts PASCAL VOC formatted object detection data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
| 
|

Format of the Segmentation data
~~~~~~~~~~~~~~~~~~~

.. warning::
    Segmentation require two kind of labels. 
    ``PNG`` files and ``class_map.txt`` 

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

**Format of label files** : ReNomIMG only accepts ``txt`` and ``PNG`` formatted label files.
The format of txt file is bellow.

Please Save as class_map.txt.

**Put file here**:``<ReNomIMG dir>/datasrc/label/segmentation/class_map.txt``

* Class number id must be start 0 and  set background
* Class munber id must be serial number

Good example

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

:red:`Bad example`

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

Sample of Segementation  PNG label file

.. image:: /_static/image/009592.png


ReNomIMG accepts PASCAL VOC formatted object detection data.

| **The PASCAL Visual Object Classes**
| http://host.robots.ox.ac.uk/pascal/VOC/
| 
|

.. note:: 
    The name of image file and corresponded label file name have to be same.
    For example, the image file name is ``image01.jpg``, corresponded label file name
    have to be ``image01.png``.


Create Detection Model
----------------------

So far, the server and dataset are prepared. Let's build a object detection model.
For building a model, you have to specify ``dataset`` and ``hyper parameters``.

Create Dataset
​~~~~~~~~~~~~~~

For training a machine learning model, you have to prepare training dataset and validation dataset.
Training dataset is used for training model, and validation dataset is used for
evaluating a model in terms of how accurately predict data that have not used in training.

In ReNomIMG, training dataset and validation dataset will be **randomly** sampled from the data
that is in the ``datasrc`` directory.

.. image:: /_static/image/how_to_use_gui_datasrc.png

According to the above figure, you can create ``dataset`` from datasrc.
Once a dataset is created its content will never be change.

For creating a ``dataset``, please move to dataset setting modal. Following figures
guide you to the dataset page.

.. image:: /_static/image/how_to_use_gui_dataset_create_button01.png

Then following page will be appeared.

.. image:: /_static/image/how_to_use_gui_dataset_create_button02.png

As you can see, you can specify the ``dataset name``, ''description'' and ``ratio of training data``.

After filling all forms, please push the ``confirm`` button to confirm the content that 
the dataset includes.

.. image:: /_static/image/how_to_use_gui_dataset_create_button03.png

Then following graph will be appeared. You can confirm what classes are included 
in the dataset and how many tags are they.

At last, for saving the dataset, please push the ``save`` button.

You can confirm created datasets in the dataset page.
For going to the dataset page, please follow the figure below.

.. image:: /_static/image/how_to_use_gui_dataset_create_button04.png

.. image:: /_static/image/how_to_use_gui_dataset_create_button05.png

In the above figure, 2 datasets are already created. 


Hyper parameter setting
​~~~~~~~~~~~~~~~~~~~~~~~

So far you got all the materials, let's build a model and run training.
For creating a model please push the button ``Add New Model``.

.. image:: /_static/image/how_to_use_gui_model_create01.png

Then you can see a hyper parameter setting modal like following figure.

.. image:: /_static/image/how_to_use_gui_model_create02.png

As you can see in above figure, you can specify following parameters.

* **Dataset Name** ... Select the dataset for training.
* **CNN architecture** ... Select the object detection algorithm.
* **Train Whole network** ... If this is set to True, whole network weight will be trained.
* **Image size** ... Image size for training.
* **Training loop setting** ... Number of training and batch size.

.. note::

    Depending on your GPU device, larger image size or batch size causes memory overflow.

Training Model
​~~~~~~~~~~~~~~

Finishing hyper parameter settings, then **push run button to start training!**

If the training starts, model will be appeared in model list and progress bar will be shown.

.. image:: /_static/image/how_to_use_gui_model_create03.png


Uninstall ReNomIMG
------------------

You can uninstall ReNomIMG by following pip command.

.. code-block :: shell

    pip uninstall renom_img

~~~~~~~~~~~~~~~~~~~
