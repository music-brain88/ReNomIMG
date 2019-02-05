Install ReNomIMG
=================

There are 3 ways for installing ReNomIMG.


Install by pip.
~~~~~~~~~~~~~~~~

You can install ReNomIMG by ``pip`` command. This is the simplest way for installation.

For python3.5

    .. code-block:: shell

        pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.0.2-cp35-cp35m-linux_x86_64.whl

For python3.6

    .. code-block:: shell

        pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.0.2-cp36-cp36m-linux_x86_64.whl


    .. note::

        This is ``linux OS`` only. If your OS is windows or MAC that are not recommended system, 
        please install ReNomIMG from binary code.

        If you have following error,
        
        .. code-block:: shell

            ImportError: No module named '_tkinter', please install the python3-tk package

        please install python3-tk using following command.

        .. code-block:: shell

            sudo apt-get install python3-tk



Install from binary.
~~~~~~~~~~~~~~~~~~~~~

    If you install ReNomIMG from binay, Node.js is required.

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

You can use ReNomIMG using Docker.
Dockerfiles and running scripts are in the 
"ReNomIMG/docker" directory.


**Build docker image**

For building Docker image, please run
following command.

.. code-block:: shell

    cd ReNomIMG/docker
    sh build.sh

Then a docker image will be created.


**Run docker image**

For running ReNomIMG server on Docker image, 
please use the script run.sh.

.. code-block:: shell

    cd ReNomIMG/docker
    sh run.sh

The script accept some arguments.

    * -d : Path to the data source directory. This directory contains image files and label files.
    * -s : Path to the data storage directory. Sqlite DB, trained weight and pretrained weight will be arranged into this directory.
    * -p : The port number.

An Example is bellow.

.. code-block:: shell

    sh run.sh -d ../datas -s ../storage -p 8999

If no arguments are passed, directories named ``datasrc`` and ``storage`` will be created in
current directory, and the application uses ``8080`` port.

.. note ::

    This requires nvidia-docker.

**Requirements**

  - OS : Ubuntu 16.04
  - python : >=3.5
  - `ReNomDL <https://github.com/ReNom-dev-team/ReNom.git>`_ : >= 2.7

For required python packages, please refer to the `requirements.txt <https://github.com/ReNom-dev-team/ReNomIMG/blob/release/2.0/requirements.txt>`_ .
