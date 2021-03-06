# Dataset

We use public [Amazon_Grocery_and_Gourmet_Food dataset](http://deepyeti.ucsd.edu/jianmo/amazon/), [Amazon_Electronics dataset](http://snap.stanford.edu/data/amazon/) (5-core version with metadata for both datasets) and [MovieLens-1M](https://grouplens.org/datasets/movielens/) as our build-in datasets. Grocery_and_Gourmet_Food and Amazon_Electronics datasets can be generated by `Amazon_Grocery.ipynb` and `Amazon_Electronics.ipynb` respectively. These datasets are compatible with all models. MovieLens dataset is compatible with all models aside Chorus and can be found in current folder or generated by `MovieLens.ipynb`. For Chorus model, there is a specially designed dataset, which is placed in current folder or can be created by `MovieLens_for_Chorus.ipynb`.  

We describe the required files below (recommend to open `Amazon_Grocery.ipynb`, `Amazon_Electronics.ipynb` or `MovieLens_for_Chorus.ipynb` to observe the format of dataset files):



**train.csv**

- Format: `user_id	item_id	time`
- All ids **begin from 1** (0 is reserved for NaN), and the followings are the same.
- Need to be sorted in **time-ascending order** when running sequential models.



**test.csv & dev.csv**

- Format: `user_id	item_id	time	neg_items`
- The last column is the list of negative items corresponding to each ground-truth item (should not include unseen item ids beyond the `item_id` column in train/dev/test sets).
- The number of negative items need to be the same for a specific set, but it can be different between dev and test sets.
- If there is no `neg_items` column, the evaluation will be performed over all the items by default.

![dev/test data format](../log/_static/format_test.png)



**item_meta.csv** (optional)

- Format: `item_id	i_category	r_complement	r_substitute`
- Optional, only needed for some of the models (CFKG, SLRC+, Chorus, KDA).
- `i_category` is the attribute of an item, such as category, brand and so on. The values need to be discrete and finite.
- `r_complement` and `r_substitute` are the relations between items, and its value is a list of items (can be empty []). 

![meta data format](../log/_static/format_meta.png)



You can also implement a new reader class based on [BaseReader.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseReader.py) and read data in your own style, as long as the basic information is included. Then assign your model with the new reader and begin to use new members of the reader when preparing batches in the model.

