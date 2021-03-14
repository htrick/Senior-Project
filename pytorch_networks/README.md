## Setup

It is best to first setup a Python virtual environment before running PyTorch.  A virtual environment will download all the Python packages into a local environment directory and prevent those packages from affecting your local Python installation.  Use the following command to setup your virtual environment (you will need to do this only once, and it will create the directory ``pt-env``):

```
python3 -m venv pt-env
```

Once your environment is setup, you will need to activate it each time you want to use that local Python installation:

```
source pt-env/bin/activate
```

Then, install the necessary packages:

```
pip3 install torch torchsummary torchvision albumentations timm
```

## Working with the training/testing scripts

To train the model run

```
python3 train.py
```

To run the inference script (which will create new images in the ``Inference_Images`` directory):

```
python3 inference model_name.pt
```


