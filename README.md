HIOB
====
HIOB is modular hierarchical object tracking framework written in python and tensorflow. It uses a combination of offline trained CNNs for visual feature extraction and online trained CNNs to build a model of the tracked object. 

HIOB was created for a diploma thesis on CNNs at the Department of Informatics of the [Universität Hamburg](https://www.uni-hamburg.de/) in the research group [Knowledge Technology (WTM)](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/). During a bachelor thesis the performance has been further improved and it has been adapted to run inside of [ROS](http://www.ros.org/).

The ROS integration of HIOB lives in a separate repository: https://github.com/theCalcaholic/hiob_ros
TODO: add link to the docker image of hiob_ros.

The algorithm in HIOB is inspired by the [FCNT](https://github.com/scott89/FCNT) by *Wang et al* presented in their [ICCV 2015 paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf). The program code of HIOB is completely independet from the FCNT and has been written by us.

# Installation

#### Using HIOB

    (hiob_venv)  python hiob_gui.py -e config/environment.yaml -t config/tracker.yaml


clone the repositiory

    $ git clone https://github.com/kratenko/HIOB.git

or

    $ git clone git@github.com:kratenko/HIOB.git

#### virtual environment
HIOB needs python3 and tensorflow. We recommend building a virtual environment for HIOB.
Build a virtual environment somewhere outside of the HIOB directory and activate it:

    $ virtualenv -ppython3 hiob_env
    $ source hiob_env/bin/activate
    
#### Installing CUDA
In order to run the gpu version, cuda needs to be installed on the machine. In order to install cuda and cudnn, perform the following actions:
1. <b>Install cuda</b> with your method of choice from <a href="https://developer.nvidia.com/cuda-downloads">here</a> 
(or <a href="https://developer.nvidia.com/cuda-toolkit-archive">older versions</a>) <br>
Theoretically, Tensorflow >= 1.11 should recogniue CUDA 10.0, but in my case it didn't hence I installed cuda 9.0 which, even though not officially suported,
runs on Ubuntu 18.04.<br>
I had the best experience installing cuda via the deb file, but every method should work. Make sure to apt-get --purge remove any previous installations, as this can be
a bit tricky though, especially if you want to install the custom graphics driver, i highly encourage anyone to read the <a href="http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf">official liunx installation guide</a>.

2. <b>Install cudnn</b> from <a href="https://developer.nvidia.com/cudnn">here</a>, you have to register a nvidia developer account in the process.
 Follow the installation instructions from <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">here</a> for a smooth installation, for me 
 the installation via tar file worked great.
 
3. <b>Add cudnn to the virtualenv path.</b> Maybe this was just buggy for me, but after successful installation of cuda 9.0 and cudnn, tensorflow would not find my
cudnn installation. Therefor go to your virtualenv python installation and add the following line to your activate file in /path_to_venv/bin/activate, right under the export PATH statement

```
    export PYTHONPATH="/usr/local/cuda-9.0/lib64"
    export PYTHONPATH="/usr/local/cuda/lib64"
```

Helpfull links
https://github.com/tensorflow/tensorflow/issues/20271
https://github.com/tensorflow/tensorflow/issues/22940
https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04


# for using your GPU and CUDA
    (hiob_env) $ cd HIOB
    (hiob_env) $ pip install -r requirements.txt 

    
#### dependencies
Install required packages:

    # for using your GPU and CUDA
    (hiob_env) $ cd HIOB
    (hiob_env) $ pip install -r requirements.txt 

This installs a tensorflow build that requires a NVIDIA GPU and the CUDA machine learning library. You can alternatively use a tensorflow build that only uses the CPU. It should work, but it will not be fast. We supply a diffenrent requirements.txt for that:

    # alternatively for using your CPU only:
    (hiob_env) $ cd HIOB
    (hiob_env) $ pip install -r requirements_cpu.txt
    

# Run the demo
HIOB comes with a simple demo script, that downloads a tracking sequence (~4.3MB) and starts the tracker on it. Inside your virtual environment and inside the HIOB directory, just run:

    (hiob_env) $ ./run_demo.sh
    
If all goes well, the sample will be downloaded to `HIOB/demo/data/tb100/Deer.zip` and a window will open that shows the tracking process. A yellow rectangle will show the position predicted by HIOB and a dark green frame will show the ground truth included in the sample sequence. A log of the tracking process will be created inside `HIOB/demo/hiob_logs` containing log output and an analysis of the process.


# Getting more test samples
## The tb100 online tracking benchmark
The deer example used in the demo is taken from the tb100 online benchmark by *Yi Wu* and *Jongwoo Lim* and *Ming-Hsuan Yang*. The benchmark consists of 98 picture sequences with a total of 100 tracking sequences. It can be found under http://visual-tracking.net/ HIOB can read work directly on the zip files provided there. The benchmark has been released in a paper:  http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf

Since the 98 sequences must be downloaded individually from a very slow server, the process is quite time consuming. HIOB comes with a script that can handle the download for you, it is located at `bin/hiob_downloader` within this repository. If you call it with argument `tb100` it will download the whole dataset from the server. This will most likely take several hours.

## The Princeton RGBD tracking benchmark
HIOB also works with the [Princeton Tracking Benchmark](http://tracking.cs.princeton.edu) and is able to read the files provided there. That benchmark provides depth information along with the RGB information, but the depth is not used by HIOB. Be advised that of the 100 sequences provided only 95 contain a ground truth. The original implementation of HIOB has been evaluated by the benchmark on April 2017, the results can be seen on the [evaluation page](http://tracking.cs.princeton.edu/eval.php) named `hiob_lc2`.
