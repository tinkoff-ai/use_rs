# User-specific embeddings in time-aware sequential recommender models

Code Release for submission "User-specific embeddings in time-aware sequential
recommender models". The repository contains source code for reproducing the experiments presented in the paper.

## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository

```bash
git clone https://github.com/tinkoff-ai/use_rs
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```

4. Run jupyter notebooks `Amazon_Grocery.ipynb` and `Amazon_Electronics.ipynb` in [data](https://github.com/tinkoff-ai/use_rs/tree/main/data) folder to download and build Grocery_and_Gourmet_Food and Electronics datasets.

5. Run the experiments with baseline models, original and modified implementations of sequential models with the build-in dataset. Each implemented model requires unique set of hyperparameters. Here are some examples:
- POP
```bash
python main.py --model_name POP --train 0 --dataset ml-1m
```

- BPRMF
```bash
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-5 --dataset ml-1m
```

- KDA:
```bash
# ORIGINAL
python main.py --model_name KDA --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 \
    --history_max 20 --dataset 'ml-1m'

# MODIFIED
python main.py --model_name KDA2 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 \
    --history_max 20 --dataset 'ml-1m'

```

- Chorus:
```bash
# ORIGINAL
!python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 \
    --batch_size 512 --dataset 'ml-1m_Chorus' --stage 1
!python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 \
    --dataset 'ml-1m_Chorus' --base_method 'BPR' --stage 2

# MODIFIED
!python main.py --model_name Chorus2 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 \
    --batch_size 512 --dataset 'ml-1m_Chorus' --stage 1
!python main.py --model_name Chorus2 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 \
    --dataset 'ml-1m_Chorus' --base_method 'BPR' --stage 2

```

- SLRC:
```bash
# ORIGINAL
python main.py --model_name SLRCPlus --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset 'ml-1m'

# MODIFIED
python main.py --model_name SLRCPlus2 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset 'ml-1m'

```

- HGN:
```bash
# ORIGINAL
python main.py --model_name HGN --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'ml-1m'

# MODIFIED
python main.py --model_name HGN2 --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'ml-1m'

```


- SHAN:
```bash
# ORIGINAL
python main.py --model_name SHAN --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 \
    --dataset 'ml-1m'

# MODIFIED
python main.py --model_name SHAN2 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 \
    --dataset 'ml-1m'

```

## Experiments

We conducted experiments to compare five original models: SHAN, HGN, SLRC, Chorus and KDA with modified models: $SHAN_{our}$, $HGN_{our}$, $SLRC_{our}$, $Chorus_{our}$ and $KDA_{our}$ on three datasets: MovieLens, Grocery_and_Gourmet_Food and Electronics.
Table bellow demonstrates main results of experements.

![experiments](https://github.com/tinkoff-ai/use_rs/blob/main/log/_static/experiments.png)




