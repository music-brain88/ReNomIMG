Quick start with example dataset
================================

Let's use ReNomIMG with an example dataset.

We will use the PASCAL VOC dataset in this example.

| **The PASCAL Visual Object Classes (VOC) Challenge**
| Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
| International Journal of Computer Vision, 88(2), 303-338, 2010
| PDF: http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf
| Web site: http://host.robots.ox.ac.uk/pascal/VOC/
|
|

Prepare Dataset
----------------

First we need to prepare the image dataset. Here we use the ``Pascal VOC`` dataset.
You can download and prepare the dataset by running the following command.


.. code-block :: shell

    # This command will download the data and save it to the current directory.
    renom_img setup_example

    # ### Setup a example dataset ###
    # 1/7: Downloading voc dataset.
    # 2/7: Extracting tar file.
    # 3/7: Moving image data to datasrc/img...
    # 4/7: Moving image data to datasrc/prediction_set/img...
    # 5/7: Moving xml data to datasrc/label/detection...
    # 6/7: Moving segmentation target data to datasrc/label/segmentation...
    # 7/7: Creating classification target data in datasrc/label/classification...
    # Setup done.


The ``datasrc`` and ``storage`` directories will be created in the current directory.

Run ReNomIMG
-------------

Once the dataset is ready, start the ReNomIMG server.
The command below runs ReNomIMG on port 8080.

.. code-block :: shell

    renom_img --port 8080

Next, let's move on to the section "How to use the ReNomIMG GUI".
