# The dataset is from [here](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

if this repo doesnt contain dataset u need to download it and put the dataset in the data folder, so the folder structure will be like this

```
├───data
│   ├───features
│   ├───preprocessed
│   │   ├───test
│   │   │   ├───benign
│   │   │   └───malignant
│   │   └───train
│   │       ├───benign
│   │       └───malignant
│   └───raw
│       ├───test
│       │   ├───benign
│       │   └───malignant
│       └───train
│           ├───benign
│           └───malignant
```

where raw is the original data, preprocessed is the data that has been preprocessed, and features is the data that has been extracted the features

# What funcion of each file?

- `preprocess.py` is file with tool / functions used to preprocess the data
- `featureextraction.py` is file with tool / functions used to extract the features from the data
- `util.py` is file with tool / functions used to help the other files
- `preprocessing.ipynb` is file where i explore the data and do the preprocessing
- `featureextraction.ipynb` is file where i explore the data and do the feature extraction
- `full.ipynb` is file where i do the training and testing
- `visualize.ipynb` is file where i do the visualization
- `final.ipynb` is the final file (you can run this file to see the result and some visualization)
