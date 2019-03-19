Quick start with example dataset
================================

Let's use ReNomIMG with example dataset.

We use PASCAL VOC dataset as an example.

| **The PASCAL Visual Object Classes (VOC) Challenge**
| Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
| International Journal of Computer Vision, 88(2), 303-338, 2010
| PDF: http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf
| Web site: http://host.robots.ox.ac.uk/pascal/VOC/
|
|

Prepare Dataset
----------------

First we need to prepare image dataset. Here we use ``Pascal VOC dataset`` .
You can setup the data only running following command.


.. code-block :: shell

    # This command will download the data and align it to current directory.
    renom_img setup_example

    # ### Setup a example dataset ###
    # 1/7: Downloading voc dataset.
    # 2/7: Extracting tar file.
    # 3/7: Moving image data to datasrc/img...
    # 4/7: Moving image data to datasrc/prediction_set/img...
    # 5/7: Moving xml data to datasrc/label/detection...
    # 6/7: Moving segmentation target data to datasrc/label/segmentation...
    # 7/7: Creating classification target data to datasrc/label/classification...
    # Setup done.


You will see the ``datasrc`` and ``storage`` directory that are created to current directory.

Run ReNomIMG
-------------

After the data preparation, run the ReNomIMG server.
Following code runs ReNomIMG with port 8080.

.. code-block :: shell

    renom_img --port 8080

Next, let's move on next settion "How to use ReNomIMG GUI".
