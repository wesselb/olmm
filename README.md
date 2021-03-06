# OLMM

Reference implementation of the Orthogonal Linear Mixing Model (OLMM)

## Installation

Python 3.6 or higher is required.
To begin with, clone and enter the repo.

```bash
git clone https://github.com/wesselb/olmm
cd olmm
```

Then make a virtual environment and install the requirements.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Finally, download the data for the example, [which consists of daily 
temperature measurements across Croatia from 2006](https://spatial-analyst.net/book/HRtemp2006).

```bash
sh fetch_data.sh
```


## Reference Implementation

A basic reference implementation of the OLMM can be found in `olmm.py`.
It illustrates how to do training, inference, and prediction.
These functions are used with AutoGrad, TensorFlow, and PyTorch as 
backend in `example_autograd.py`, `example_tensorflow.py`, and
`example_pytorch.py`, respectively.
The three examples execute the same task of loading and preparing the data,
fitting a simple OLMM, making predictions and plotting some of the latent 
processes and some of the outputs.
