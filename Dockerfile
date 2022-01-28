FROM debian:buster

RUN apt-get update

# set noninteractive installation
RUN export DEBIAN_FRONTEND=noninteractive
# install tzdata package
RUN apt-get install -y tzdata
# set your timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y \
    bzip2 \
    g++ \
    git \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-opencv

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -r ./requirements.txt

RUN apt-get autoremove -y; apt-get autoclean -y

COPY cpop ./cpop
