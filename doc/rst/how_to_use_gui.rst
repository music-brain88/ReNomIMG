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
      ├── label # Set training label files here.
      └── prediction_set
            ├── img     # Set prediction img files here.
            └── output  # Prediction result will be output here.
                  ├── csv
                  └── xml

As written in the above comments, please set training image data to ``datasrc/img``,
set training label data to ``datasrc/label``.

.. note::

    The name of image file and corresponded label file name have to be same.
    For example, the image file name is ``image01.jpg``, corresponded label file name
    have to be ``image01.xml``.


Format of the data
~~~~~~~~~~~~~~~~~~~

**Format of image files** : ReNomIMG only accepts ``JPEG`` and ``PNG`` formatted image files.

**Format of label files** : ReNomIMG only accepts ``xml`` formatted label files.
The format of xml file is bellow.


Create Detection Model
----------------------

Hyper parameter setting
~~~~~~~~~~~~~~~~~~~~~~~

Create Dataset
~~~~~~~~~~~~~~

Model Evaluation
----------------

Learning Curve
~~~~~~~~~~~~~~

IOU & mAP
~~~~~~~~~~

Use Trained Model
-----------------

Deploying
~~~~~~~~~

Prediction
~~~~~~~~~~

Uninstall ReNomIMG
------------------

You can uninstall ReNomIMG by following pip command.

.. code-block :: shell

    pip uninstall renom_img
