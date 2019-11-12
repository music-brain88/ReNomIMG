# ReNomIMG

ReNomIMG is an image recognition model development tool.

- https://www.renom.jp/packages/renomimg/index.html


## Version

v2.3.2


## Changes

v2.3b2 => v2.3.2

- Fixed bug where model ID information was displayed as 'undefined' when stopping model training on GUI
- Fixed bug in GUI Prediction Result display where 'Image' toggle status on Segmentation Page affected whether or not images were displayed on the Classification and Detection pages
- Modified error message displayed on GUI when training diverges and model parameters cause a numerical overflow to be more descriptive and suggest alternative actions to take
- Fixed incorrect batch normalization layer momentum value for Yolo v1, Yolo v2, ResNet and ResNeXt models
- Fixed bug in Python API Yolo v2 loss function that caused error when image height and width were not equal to each other

Please refer to the change log at the renom.jp URL below for a complete change history:

- https://www.renom.jp/packages/renomimg/rsts/change.html


## Recommended Environment

- OS: Ubuntu 16.04
- Browser: Google Chrome (version 63.0.3239.132)
- Python: 3.5 or 3.6
- ReNom: 2.7.3


## Requirements

ReNomIMG v2.3.2 requires ReNom version 2.7.3.
If you haven't installed ReNom, first install it from https://github.com/ReNom-dev-team/ReNom.git.

For required python modules please refer to requirements.txt.


### Important note

ReNom products can only handle filenames with alphanumeric characters and hyphens.


## Installation

ReNomIMG v2.3.2 requires ReNom 2.7.3, which can be installed from https://github.com/ReNom-dev-team/ReNom.git.


### Installing the ReNomIMG package

Linux users can install ReNomIMG from the following Wheel package.

Users on a different OS must install from source (refer to instructions below).

The Wheel package is provided at:

`https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-VERSION-cp36-cp36m-linux_x86_64.whl`

(VERSION should be replaced with the actual version number, e.g. 0.0.1)

You can install the wheel package with pip3 command::

- For Python3.5
`pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.3.2-cp35-cp35m-linux_x86_64.whl`

- For Python3.6
`pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-2.3.2-cp36-cp36m-linux_x86_64.whl`


#### Wheels for past versions

For python 3.5
- [v2.3b2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b2-cp35-cp35m-linux_x86_64.whl)
- [v2.3b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b1-cp35-cp35m-linux_x86_64.whl)
- [v2.3b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b0-cp35-cp35m-linux_x86_64.whl)
- [v2.2b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.2b1-cp35-cp35m-linux_x86_64.whl)
- [v2.2b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.2b0-cp35-cp35m-linux_x86_64.whl)
- [v2.1b3](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b3-cp35-cp35m-linux_x86_64.whl)
- [v2.1b2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b2-cp35-cp35m-linux_x86_64.whl)
- [v2.1b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b1-cp35-cp35m-linux_x86_64.whl)
- [v2.1b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b0-cp35-cp35m-linux_x86_64.whl)
- [v2.0.4](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.4-cp35-cp35m-linux_x86_64.whl)
- [v2.0.3](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.3-cp35-cp35m-linux_x86_64.whl)
- [v2.0.2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.2-cp35-cp35m-linux_x86_64.whl)
- [v2.0.1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.1-cp35-cp35m-linux_x86_64.whl)
- [v2.0.0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.0-cp35-cp35m-linux_x86_64.whl)

For python 3.6
- [v2.3b2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b2-cp36-cp36m-linux_x86_64.whl)
- [v2.3b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b1-cp36-cp36m-linux_x86_64.whl)
- [v2.3b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.3b0-cp36-cp36m-linux_x86_64.whl)
- [v2.2b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.2b1-cp36-cp36m-linux_x86_64.whl)
- [v2.2b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.2b0-cp36-cp36m-linux_x86_64.whl)
- [v2.1b3](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b3-cp36-cp36m-linux_x86_64.whl)
- [v2.1b2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b2-cp36-cp36m-linux_x86_64.whl)
- [v2.1b1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b1-cp36-cp36m-linux_x86_64.whl)
- [v2.1b0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.1b0-cp36-cp36m-linux_x86_64.whl)
- [v2.0.4](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.4-cp36-cp36m-linux_x86_64.whl)
- [v2.0.3](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.3-cp36-cp36m-linux_x86_64.whl)
- [v2.0.2](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.2-cp36-cp36m-linux_x86_64.whl)
- [v2.0.1](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.1-cp36-cp36m-linux_x86_64.whl)
- [v2.0.0](https://renom.jp/docs/downloads/wheels/renom_img/renom_img-2.0.0-cp36-cp36m-linux_x86_64.whl)


### Installing from source

For installing ReNomIMG from source, download the repository from the following URL.

`git clone https://github.com/ReNom-dev-team/ReNomIMG.git`

Next, move into the ReNomIMG directory.

`cd ReNomIMG`

First install all required packages via pip.

`pip install -r requirements.txt`

Next, install the ReNomIMG module using the following command.

`pip install -e .`

Finally, build the extension modules.

`python setup.py build`


## How to use

Please refer to the following link for detailed instructions: 

- ReNomIMG - renom.jp

http://renom.jp/packages/renomimg/index.html


### Quick start - Start with example data.

The following command prepares an example dataset [PASCAL VOC].
You can try ReNomIMG right away with this example dataset.

`renom_img setup_example`

This command will create the `datasrc` and `storage` directories.

### Quck start - How to start

Type the following command in the ReNomIMG directory.

`python -m renom_img`

Alternatively, use the following command (available from version 0.7 beta onward).

`renom_img`

The second command can be called from any directory and creates the `datasrc` folder in the current directory.
Please place your dataset inside the created directory.

After the server starts, you should see a message like the following.

<img src='./js/static/img/server_run.png' width='60%'/>


## License

“ReNomIMG” is provided by GRID inc., as subscribed software.  By downloading ReNomIMG, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNomIMG for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNomIMG for academic, educational and/or product evaluation purposes, you may use ReNomIMG royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.

[PASCAL VOC]:http://host.robots.ox.ac.uk/pascal/VOC/
