#!/bin/sh -x

# Movie Lens
python main.py --model_name POP --train 0 --dataset ml-1m

#Amazon Electronics 
python main.py --model_name POP --train 0 --dataset Amazon_Electronics

#Grocery and Gourmet Food 
python main.py --model_name POP --train 0 --dataset Grocery_and_Gourmet_Food_1m