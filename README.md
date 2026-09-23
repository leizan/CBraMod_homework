# CBraMod and EEGSimpleConv on SHU-MI

The main file is **CBraMod.ipynb**. It contains the paper discussion, model training, results comparison, data-efficiency experiments, and inference-time measurements.

The current configurations use **2 epochs and 5 seeds**. These results should be considered preliminary.

## 1. Python environment

The local experiments used **Python 3.11**. Install the required packages from the project directory:

```bash
python -m pip install -r requirements.txt
```


## 2. SHU-MI dataset

Download the MAT-format SHU-MI dataset from the [official dataset page](https://figshare.com/articles/code/shu_dataset/19228725).

Store the downloaded archive in `data/19228725/` and extract the `.mat` files into:

```text
data/19228725/mat/
```

To preprocess the data, update the input and output paths in `preprocessing/preprocessing_shu.py` to match your local project location:

```python
root_dir = '/Users/leizan/Desktop/SigmaNova/data/19228725/mat'

db = lmdb.open(
    '/Users/leizan/Desktop/SigmaNova/data/preprocessed',
    map_size=4 * 1024**3,
)
```

Then run the following commands from the project directory:

```bash
mkdir -p data/preprocessed
python preprocessing/preprocessing_shu.py
```

The script resamples each four-second EEG trial to 800 time points, reshapes it to `[32, 4, 200]`, converts the labels to 0 and 1, and saves the processed LMDB database in `data/preprocessed/`.


## 3. Pretrained model weights

Place the CBraMod pretrained checkpoint at:

```text
models/model_weights/CBraMod/pretrained_weights.pth
```

The checkpoint is available from [Hugging Face](https://huggingface.co/weighting666/CBraMod).

EEGSimpleConv is trained from scratch and does not require pretrained weights.

## 4. Run the notebook

Open **CBraMod.ipynb** and update the project path in the first code cell:

```python
PROJECT = Path("/Users/leizan/Desktop/SigmaNova")
```

Replace this path with your own project location, then run the cells in order.

Choose `configs/CBraMod.json` or `configs/EEGSimpleConv.json` to select the model. Training settings, including epochs, batch size, and output directory, are defined in these configuration files.


