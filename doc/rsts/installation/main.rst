Install ReNomIMG
=================

There are 3 ways to install ReNomIMG.


Install by pip.
~~~~~~~~~~~~~~~~

You can install ReNomIMG with the ``pip`` command. This is the simplest way for installation.

For python3.5

    .. code-block:: shell

        pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.1b3-cp35-cp35m-linux_x86_64.whl

For python3.6

    .. code-block:: shell

        pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.1b3-cp36-cp36m-linux_x86_64.whl


    .. note::

        This command is for ``Linux OS`` environments only. If your OS is Windows or Mac, which are not recommended for ReNomIMG,
        please install ReNomIMG from binary code according to the instructions below.

        If you get the following error,
        
        .. code-block:: shell

            ImportError: No module named '_tkinter', please install the python3-tk package

        please install python3-tk using the following command.

        .. code-block:: shell

            sudo apt-get install python3-tk



Install from binary.
~~~~~~~~~~~~~~~~~~~~~

    To install ReNomIMG from binary, Node.js is required.

    .. code-block:: shell

        git clone https://github.com/ReNom-dev-team/ReNomIMG.git
        cd ReNomIMG/
        pip install -r requirements.txt
        python setup.py build
        pip install -e .

    .. note ::

        This requires ``node.js``.

Docker Image.
~~~~~~~~~~~~~~

You can use ReNomIMG via Docker.
Dockerfiles and running scripts are in the 
"ReNomIMG/docker" directory.


**Build docker image**

For building a Docker image, please run
following command.

.. code-block:: shell

    cd ReNomIMG/docker
    sh build.sh

This will create the docker image.


**Run docker image**

For running the ReNomIMG server on a Docker image,
please use the script run.sh.

.. code-block:: shell

    cd ReNomIMG/docker
    sh run.sh

This script accepts the following arguments.

    * -d : Path to the data source directory. This directory contains image files and label files.
    * -s : Path to the data storage directory. Sqlite DB, trained weights and pretrained weights will be stored in this directory.
    * -p : The port number.

An example is shown below.

.. code-block:: shell

    sh run.sh -d ../datas -s ../storage -p 8999

If no arguments are passed, directories named ``datasrc`` and ``storage`` will be created in
the current directory, and the application will use port number ``8080``.

.. note ::

    This requires nvidia-docker.

**Requirements**

  - OS : Ubuntu 16.04
  - python : >=3.5
  - `ReNomDL <https://github.com/ReNom-dev-team/ReNom.git>`_ : = 2.7.3

For required python packages, please refer to the `requirements.txt <https://github.com/ReNom-dev-team/ReNomIMG/blob/release/2.1/requirements.txt>`_ .
