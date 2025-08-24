The following are all the modules I've imported and what I imported them as. I used python version 3.12 to run my code.

## Dataset
I used the ML-100K dataset from the following link: https://grouplens.org/datasets/movielens/

## Modules
- **argparse**
- **numpy** (imported as `np`)
- **scipy**
  - `scipy.sparse`
  - `scipy.sparse.linalg`
  - `svds`
- **pandas**
- **math**
- **random**
- **sklearn**
  - `sklearn.preprocessing` (for `LabelEncoder` and `MinMaxScaler`)
  - `sklearn.model_selection` (for `train_test_split`)
  - `sklearn.metrics` (for `mean_squared_error` and `cosine_similarity`)
- **matplotlib.pyplot** (imported as `plt`)
- **seaborn** (imported as `sns`)
- **torch**
  - `torch.nn`
  - `torch.optim`
  - `torch.utils.data`
