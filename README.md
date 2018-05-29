
# ReNomIMG 0.7beta

ReNom IMG is model developing tool for object detection.

## Recommended Environment
- OS: Ubuntu 16.04
- Browser: Google Chrome(version 63.0.3239.132)
- Python: >=3.5

## Install
ReNomIMG requires ReNom.

If you haven't install ReNom, you must install ReNom from www.renom.jp.

## Install ReNomIMG package
Linux user can install ReNomIMG from Wheel package.

Other os user can't install from Wheel package but can install from source.

The Wheel package is provided at:

`https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-VERSION-cp35-cp35m-linux_x86_64.whl`

(VERSION is stands for actual version number e.g. 0.0.1)

You can install the wheel package with pip3 command::

`pip3 install https://grid-devs .gitlab.io/ReNomIMG/bin/renom_img-0.7b0-cp35-cp35m-linux_x86_64.whl`

## Install from source
For installing ReNomIMG, download the repository from following url.

`git clone https://github.com/ReNom-dev-team/ReNomIMG.git`

And move into ReNomIMG directory.
`cd ReNomIMG`

Then install all required packages.

`pip install -r requirements.txt`

At last, install renomimg module using following command.

`pip install -e .`


## How to start

1.Type following command in ReNomIMG directory.

`python -m renom_img`

Or, following command is available from 0.7beta.

`renom_img`

Second command can be called in any folder and creates `dataset` folder in current directory.
Please set your dataset to the created directory.


If the server starts, you will see a message like below.

<img src='./js/static/img/server_run.png' width='60%'/>

Then 'dataset' folder will be created in your current directory.  
Please set images and labels according `2.Create dataset directory` description.


## How to use

The following videos describes how to use ReNomIMG.
In this video, the Oxford-IIIT Pet Dataset is used.

- Cats and Dogs Classification
https://github.com/JDonini/Cats_Dogs_Classification

- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
Cats and Dogs
IEEE Conference on Computer Vision and Pattern Recognition, 2012
Bibtex
http://www.robots.ox.ac.uk/~vgg/data/pets/

#### 1.Download weight file of YOLO.
At first, you must install weight file of yolo from url.

You can get the file with the following command.

```
cd ReNomIMG

curl -O http://docs.renom.jp/downloads/weights/yolo.h5
```

#### 2.Create dataset directory
As following video showing, please create the dataset directory in the `ReNomIMG`.

When you finish, you can show following folder structure.

```
ReNomIMG
 │
 └ dataset
     │
     ├ train_set // training dataset
     │      │
     │      ├ img // training images
     │      └ label // training annotation xml files
     │
     ├ valid_set // validation dataset
     │      │
     │      ├ img // validation images
     │      └ label // validation annotation xml files
     │
     └ prediction_set // prediction dataset
            │
            ├ img // prediction images
            └ out // prediction result xml files
```

[![Not supported browser](http://img.youtube.com/vi/vKEtrMD0UII/0.jpg)](http://www.youtube.com/watch?v=vKEtrMD0UII)

#### 3.Set image data to ReNomIMG
As following video showing, please put the image you want to use in the `ReNomIMG/dataset` folder.

[![Not supported browser](http://img.youtube.com/vi/BfFY2cg1jjw/0.jpg)](http://www.youtube.com/watch?v=BfFY2cg1jjw)

#### 4.Run ReNomIMG
Same as before mentioned in 'How to start', following video describes how to start ReNomIMG.

[![Not supported browser](http://img.youtube.com/vi/zASwzmWLV9U/0.jpg)](http://www.youtube.com/watch?v=zASwzmWLV9U)

#### 5.How to start training in ReNomIMG
Following video describes how to start model training in ReNomIMG.

[![Not supported browser](http://img.youtube.com/vi/BzNTtdrMtIo/0.jpg)](http://www.youtube.com/watch?v=BzNTtdrMtIo)

#### 6.How to use model for prediction.
Following video describes how to use model for prediction in ReNomIMG.

[![Not supported browser](http://img.youtube.com/vi/RRW8ODUmUfE/0.jpg)](http://www.youtube.com/watch?v=RRW8ODUmUfE)


## Format of xml file.

The format of the xml file which created by ReNomTAG follows [PASCAL VOC] format.

An example is bellow.

```
<annotation>
 <folder>
  dataset
 </folder>
 <filename>
  2007_000027.jpg
 </filename>
 <object>
  <pose>
   Unspecified
  </pose>
  <name>
   cat
  </name>
  <truncated>
   0
  </truncated>
  <difficult>
   0
  </difficult>
  <bndbox>
   <ymax>
    203.02013422818794
   </ymax>
   <xmin>
    134.7902328154634
   </xmin>
   <xmax>
    238.81923552543284
   </xmax>
   <ymin>
    104.02684563758389
   </ymin>
  </bndbox>
 </object>
 <source>
  <database>
   Unknown
  </database>
 </source>
 <path>
  dataset/2007_000027.jpg
 </path>
 <segments>
  0
 </segments>
 <size>
  <width>
   486
  </width>
  <height>
   500
  </height>
  <depth>
   3
  </depth>
 </size>
</annotation>
```

## License

“ReNomIMG” is provided by GRID inc., as subscribed software.  By downloading ReNomIMG, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNomIMG for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNomIMG for academic, educational and/or product evaluation purposes, you may use ReNomIMG royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.

[PASCAL VOC]:http://host.robots.ox.ac.uk/pascal/VOC/
