
# Eric The composer

Eric is an A.I. bot who generates midi music files using LSTM

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![h5py 2.6.0](https://img.shields.io/badge/h5py-2.6.0-blue.svg)](https://docs.h5py.org/en/stable/build.html)
[![TensorFlow 2.1.0](https://img.shields.io/badge/TensorFlow-2.3.1-blue.svg)](https://www.tensorflow.org/install)
[![music21 5.7.2](https://img.shields.io/badge/music21-5.7.2-blue.svg)](https://github.com/cuthbertLab/music21/releases/)
[![numpy 1.18.5](https://img.shields.io/badge/numpy-1.18.5-blue.svg)](https://numpy.org/doc/stable/user/whatisnumpy.html)


## Deployment

Setup workspace for training the model

```bash
 docker pull nvcr.io/nvidia/tensorflow:20.03-tf2-py3
 docker run --gpus all -it -v /home/$USER:/workspace nvcr.io/nvidia/tensorflow:20.03-tf2-py3
 git clone git@github.com:siddarthreddygsr/eric-on-github.git
```

Installing dependencies
```bash
 pip3 install -r requirements.txt
```
Training the model

```bash
 python3 eric2.py
 ```
Getting Results
```bash
 python3 ericCompose.py
 ```


## Folder Structure

<p>The training data is stored in <code style="background-color:rgba(0, 0, 0, 0.0470588);"><a>./training-data</a></code></p>
<p>The data processed by music21 is stored in <code style="background-color:rgba(0, 0, 0, 0.0470588);"><a>./pickle/data</a></code></p>
<p>The models will be stored in <code style="background-color:rgba(0, 0, 0, 0.0470588);"><a>./models</a></code>
