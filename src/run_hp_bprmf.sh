#!/bin/sh -x

# Movie Lens
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-5 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-4 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-3 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-2 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-5 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-4 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-3 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-2 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-6 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-5 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-4 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-3 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-2 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-6 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-5 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-4 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-3 --dataset ml-1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-2 --dataset ml-1m

#Amazon Electronics 
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-5 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-4 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-3 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-2 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-5 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-4 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-3 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-2 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-6 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-5 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-4 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-3 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-2 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-6 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-5 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-4 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-3 --dataset Amazon_Electronics
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-2 --dataset Amazon_Electronics

#Grocery and Gourmet Food 
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-5 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-4 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-3 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 32 --lr 1e-3 --l2 1e-2 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-5 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-4 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-3 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-2 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-5 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-4 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-3 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 128 --lr 1e-3 --l2 1e-2 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-5 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-4 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-3 --dataset Grocery_and_Gourmet_Food_1m
python main.py --model_name BPRMF --emb_size 256 --lr 1e-3 --l2 1e-2 --dataset Grocery_and_Gourmet_Food_1m