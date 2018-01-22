---
layout: post
title:  "GPU instance in GCP"
date:   2018-01-12 00:00:00 +0900
categories: gcp
fbcomments: true
---
 
## GPU instance in GCP

* [Adding GPUs to Instances](https://cloud.google.com/compute/docs/gpus/add-gpus)

1. Install Anaconda
```bash
curl -O http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-8.0.61-1.x86_64.rpm
sudo yum install bzip2
bash Anaconda3-5.0.1-Linux-x86_64.sh
source .bashrc

conda create -n py3 python=3.6
```

2. Jupyter
```bash
sudo yum install git -y
mkdir git
cd git
git clone https://github.com/byam/dlnd.git
pip install -r requirements.txt


# server
sudo yum install tmux
tmux n -s dlnd
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

##  detach session (ctr + b, d) 
##  attach session (tmux a -t dlnd) 


# local
gcloud compute ssh instance-1 --project byam-kaggle-dev --zone us-east1-d --ssh-flag="-L" --ssh-flag="2222:localhost:8888"
```