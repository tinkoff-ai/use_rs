#!/bin/sh -x

# Movie Lens
python main.py --model_name SHAN --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset ml-1m
python main.py --model_name SHAN2 --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset ml-1m

python main.py --model_name HGN --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset ml-1m
python main.py --model_name HGN2 --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset ml-1m

python main.py --model_name SLRCPlus --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset ml-1m
python main.py --model_name SLRCPlus2 --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset ml-1m

python main.py --model_name Chorus --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset ml-1m_Chorus --stage 1
python main.py --model_name Chorus --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset ml-1m_Chorus --base_method 'BPR' --stage 2
python main.py --model_name Chorus2 --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset ml-1m_Chorus --stage 1
python main.py --model_name Chorus2 --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset ml-1m_Chorus --base_method 'BPR' --stage 2

python main.py --model_name KDA --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset ml-1m
python main.py --model_name KDA2 --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset ml-1m

#Amazon Electronics 
python main.py --model_name SHAN --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset Electronics
python main.py --model_name SHAN2 --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset Electronics

python main.py --model_name HGN --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Electronics
python main.py --model_name HGN2 --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Electronics

python main.py --model_name SLRCPlus --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset Electronics
python main.py --model_name SLRCPlus2 --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset Electronics

python main.py --model_name Chorus --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset Electronics --stage 1
python main.py --model_name Chorus --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset Electronics --base_method 'BPR' --stage 2
python main.py --model_name Chorus2 --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset Electronics --stage 1
python main.py --model_name Chorus2 --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset Electronics --base_method 'BPR' --stage 2

python main.py --model_name KDA --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset Electronics
python main.py --model_name KDA2 --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset Electronics

#Grocery and Gourmet Food 
python main.py --model_name SHAN --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset Grocery_and_Gourmet_Food
python main.py --model_name SHAN2 --epoch 500 --emb_size 64 --reg_weight 0.001 --pool_type 'average' --lr 1e-3 --l2 1e-4 --history_max 20 --dataset Grocery_and_Gourmet_Food

python main.py --model_name HGN --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food
python main.py --model_name HGN2 --epoch 500 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food

python main.py --model_name SLRCPlus --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset Grocery_and_Gourmet_Food
python main.py --model_name SLRCPlus2 --epoch 500 --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset Grocery_and_Gourmet_Food

python main.py --model_name Chorus --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset Grocery_and_Gourmet_Food --stage 1
python main.py --model_name Chorus --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset Grocery_and_Gourmet_Food --base_method 'BPR' --stage 2
python main.py --model_name Chorus2 --epoch 50 --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 --batch_size 512 --dataset Grocery_and_Gourmet_Food --stage 1
python main.py --model_name Chorus2 --epoch 500 --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --dataset Grocery_and_Gourmet_Food --base_method 'BPR' --stage 2

python main.py --model_name KDA --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset Grocery_and_Gourmet_Food
python main.py --model_name KDA2 --epoch 500 --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset Grocery_and_Gourmet_Food