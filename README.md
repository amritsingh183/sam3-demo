# Sam3 Demo

A Gardio demo for SAM3


## Download the weights from HF 

The weights are [here](https://huggingface.co/facebook/sam3)


## Install the requirements

You can use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to setup the environment


```bash
uv venv ~/uvenvs/sam3 --python 3.13
source ~/uvenvs/sam3/bin/activate
uv pip install git+https://github.com/huggingface/transformers torchvision matplotlib accelerate gradio opencv-contrib-python
```

## Run the demo


```bash
source ~/uvenvs/sam3/bin/activate

python3 main.py
```

now check your terminal and go to the URL specified in the gradio output

## My HF profile

https://huggingface.co/AbacusGauge

## Example segmentation


> Finding objects in a image (snake, birds, hats)

![demo1](images/demo.png)


> Finding search box in a web page (search box)

![demo2](images/demo2.png)


# License

Check the LICENSE file