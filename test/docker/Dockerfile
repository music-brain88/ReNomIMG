FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# update apt info & install packages
RUN apt-get update \
 && apt-get install -y \
    # ssh-server
    openssh-server \
    # common
    git \
    vim \
    curl \
    wget \
    # for install python3.6.8 build packages
    build-essential \
    python3-dev \
    libsqlite3-dev \
    libreadline6-dev \
    libgdbm-dev \
    zlib1g-dev \
    libbz2-dev \
    sqlite3 \
    tk-dev \
    zip \
    libssl-dev \
    # for renom install & build
    nodejs \
    npm \
    libsm6 \
    libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# mkdir sshd
RUN mkdir /var/run/sshd

# install python3.6.8
RUN wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz \
 && tar xf Python-3.6.8.tgz \
 && cd Python-3.6.8 \
 && ./configure --prefix=/opt/python3.6.8 \
 && make \
 && make install \
 && ln -s /opt/python3.6.8/bin/python3.6 /usr/local/bin/python3.6 \
 && ln -s /opt/python3.6.8/bin/pip3.6 /usr/local/bin/pip

# install n
RUN npm install -g n

# install node stabled version
# node 11.x.x is unusable
RUN n 10.15.3

# useradd
RUN useradd -m -s /bin/bash test
RUN echo 'test:testpassword' | chpasswd
USER test
WORKDIR /home/test

# export env
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV RENOM_PRECISION=32

# install ReNomDL
RUN git clone -b v2.7.3 https://github.com/ReNom-dev-team/ReNom.git \
 && cd ReNom \
 && pip install numpy \
 && pip install -r requirements.txt \
 && python3.6 setup.py build_ext -if \
 && pip install -e .

# install ReNomIMG
ADD ReNomIMG ReNomIMG
RUN cd ReNomIMG \
 && pip install -r requirements.txt \
 && python3.6 setup.py build \
 && pip install -e .

CMD bash
