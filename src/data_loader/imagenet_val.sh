#!/bin/bash
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
echo "chnage directory to: $1"
cd $1
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
