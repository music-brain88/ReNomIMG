- FCNの論文の再現をしたい場合

1.caffe\_to\_renom\_fcn.py
FCNの著者がcaffeでVGG-16のモデルのweightを使って更にVOC2012に対してfine tuningしたモデルが公開されています。
https://github.com/shelhamer/fcn.berkeleyvision.org
実行するためにはexternalディレクトリをmkdirしてその中に上記のリポジトリをクローンしてきてください。
必要なファイルが格納されています。
python caffe\_to\_renom\_fcn.py
を実行することで、./data/models/caffeにcaffeのモデルがダウンロードされてきます。
そのあとに./data/models/renomにコンバートされたweightが保存されます。
weightを変換するときにweightで-1を使って反転させている箇所があると思いますが、ReNomは他のフレームワークとConvolutionとDeconvolutionの掛ける順番が異なるため、この順番に変換しなければいけません。

2.evaluation.py
python evaluation.py ./data/model/renom/fcn8s\_from\_caffe.h5
などのように最後に変換したReNom用のweightを指定します。
これで論文の再現ができるはずです。
データセットについては./VOCdevkit/VOC2012/となるようにダウンロードを公式ページや関連ページからしてきてください。

- 独自のデータセットで学習をさせたい場合

上記の方法では、論文の再現はできますが、提供されているweightが既にVOC2012用にfine tuningされたデータセットですので、実際にお客さんが自分のデータで学習したいといった場合にはVGG-16のweightを用いて自分のデータ用のfine tuningをしたいと思います。

1.caffe\_to\_renom\_vgg.py
https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
VGG16のweightも元々はcaffeで提供されているweightを使用します。(Model Zoo)
公開されているprototxtは上記のリンクにありますので、./external/VGG\_ILSVRC\_16ディレクトリにソースコード中にダウンロードしてくるようになっています。
もし、ダウンロードが上手くいかなければ、リンク先からダウンロードしてきてください。
こちらも実行が完了すると、./data/models/vgg16\_from\_caffe.h5として重みが作成されるはずです。

2.fine\_tuning.py
Camvid用のVGGのfine tuning用のファイルとなっています。
ただ、最終的に作成したものはチュートリアルとして公開しています。
確かこのコードは完成では無かったので、参考程度に残しておきますが、チュートリアルと差分があれば、おそらくその箇所はバグとなっているはずです。
Camvidを./CamVidとしてダウンロードしてください。
