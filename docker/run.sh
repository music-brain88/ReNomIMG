
FLG_A="FALSE"
FLG_B="FALSE"
FLG_C="FALSE"
PORT=8080

while getopts ab:c: OPT
do
  case $OPT in
    "v" ) FLG_A="TRUE" ; DATA_PATH="$OPTARG";;
    "s" ) FLG_B="TRUE" ; STORAGE_PATH="$OPTARG" ;;
    "p" ) PORT="$OPTARG" ;;
  esac
done

if [ "$FLG_A" = "FALSE" ]; then
    DATA_PATH=$(pwd)/datasrc
    if [ ! -e $DATA_PATH ]; then
        mkdir -p $DATA_PATH
    fi
fi

if [ "$FLG_B" = "FALSE" ]; then
    STORAGE_PATH=$(pwd)/storage
    if [ ! -e $STORAGE_PATH ]; then
        mkdir -p $STORAGE_PATH
    fi
fi

echo "RUNNING Docker image"
echo "PORT: " $PORT
echo "DATA PATH: " $DATA_PATH
echo "STORAGE PATH: " $STORAGE_PATH

nvidia-docker run -d -p $PORT:8080 -v $DATA_PATH:/var/datasrc -v $STORAGE_PATH:/var/storage renom_img_docker
