
# ReNomIMG

ReNom IMG is model developing tool for object detection.

## Install
ReNomIMG requires ReNom.

If you haven't install ReNom, you must install ReNom from www.renom.jp.

For installing ReNomIMG, download the repository from following url.

`git clone https://gitlab.com/suwa/ReNomIMG.git`

And move into ReNomIMG directory.
`cd ReNomIMG`

Then install all required packages.

`pip install -r requirements.txt`


## How to start

1.Move to ReNomIMG directory using following command.

`cd ReNomIMG`

2.Run server.py script and the application server starts.

`python server.py`

If the server starts, you will see a message like below.

<img src='./static/img/server_run.png' width='60%'/>

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
As following video showing, please create the dataset direcotry in the `ReNomIMG`.

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

*The introduction movie will be uploaded later.
Currently the video is in the folder `static/movie/movie01.mp4`.

#### 3.Set image data to ReNomIMG
As following video showing, please put the image you want to use in the `ReNomIMG/dataset` folder.

*The introduction movie will be uploaded later.
Currently the video is in the folder `static/movie/movie02.mp4`.

#### 4.Run ReNomIMG
Same as before mentioned in 'How to start', following video describes how to start ReNomIMG.

*The introduction movie will be uploaded later.
Currently the video is in the folder `static/movie/movie03.mp4`.

#### 5.How to start training in ReNomIMG
Following video describes how to start model training in ReNomIMG.

*The introduction movie will be uploaded later.
Currently the video is in the folder `static/movie/movie04.mp4`.

#### 6.How to use model for prediction.
Following video describes how to use model for prediction in ReNomIMG.

*The introduction movie will be uploaded later.
Currently the video is in the folder `static/movie/movie05.mp4`.

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
