#!/usr/bin/env bash

mkdir -p datasets

cd datasets

#echo "Downloading birds dataset ..."
#if [ ! -f cub-2011.tgz ]
#then
#	wget -O cub-2011.tgz http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
#fi
#mkdir -p cub-2011
#tar -xzf cub-2011.tgz -C cub-2011 --strip-components=1
#
#echo "Downloading CIFAR 100 dataset ..."
#
#if [ ! -f cifar-100-python.tar.gz  ]
#then
#	wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
#fi
#mkdir cifar-100
#tar -xzf cifar-100-python.tar.gz -C cifar-100 --strip-components=1

echo "Downloading products dataset ..."
if [ ! -f Stanford_Online_Products.zip ]
then
	wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
fi
unzip Stanford_Online_Products.zip

cd ..
