#!/bin/bash
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r requirements.txt

wandb login 5b6135a90b6a8579b77079cb2dd5848a67b25e6e