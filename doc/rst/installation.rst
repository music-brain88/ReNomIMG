Install ReNomIMG
=================

ReNomIMG requires following python modules.


Install by pip.
~~~~~~~~~~~~~~~~

You can install ReNomIMG by ``pip`` command. This is the simplest way for installation.


    .. code-block:: shell

        pip install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-0.9b0-cp35-cp35m-linux_x86_64.whl


    .. note::

        This is ``linux OS`` only. If your OS is windows or MAC, please install ReNomIMG
        from binary code.


Install from binary.
~~~~~~~~~~~~~~~~~~~~~

    If you install ReNomIMG from binay, Node.js is required.

    .. code-block:: shell

        git clone ~~
        cd ReNomIMG/
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
please use the script `run.sh`.

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
