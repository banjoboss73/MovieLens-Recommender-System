import argparse
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


df_basic = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
info = pd.read_csv('ml-100k/u.info', sep='\t', header=None)


columns = [ 'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ]
items = pd.read_csv('ml-100k/u.item', sep='|', header=None, names=columns, encoding='ISO-8859-1') 



# In[4]:


genre = pd.read_csv('ml-100k/u.genre', sep='|', header=None)
genre


# In[5]:


user  = pd.read_csv('ml-100k/u.user', sep='|', header=None, names=['user_id','age','gender','occupation','zip code'])



# ## Preprocessing the data

# In[6]:


df_1 = pd.merge(df_basic, user, on='user_id', how='inner')



# In[7]:


df = pd.merge(df_1, items, left_on='item_id', right_on='movie_id',how='inner')



# In[8]:




# ## Cleaning the dataset

# In[9]:


# drop null values
df = df.dropna(axis=1, how='all')



# In[10]:


df = df.dropna()



# In[11]:


df['unknown'].value_counts()


# In[12]:


# Since the 'unknown' has no values, we can drop it
df = df.drop(columns=['unknown'])


# In[13]:


# Checking for duplicates
df.duplicated().sum()


# In[14]:


# Convert the timestamps to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')



# In[15]:


zip_codes = pd.read_csv("ml-100k/ZIP-COUNTY-FIPS_2017-06.csv")




# In[16]:





# In[17]:





# In[18]:


# check for duplicate zip codes
zip_codes['ZIP'].duplicated().sum()

# removing duplicates
zip_codes = zip_codes.drop_duplicates(subset=['ZIP'])
df['zip code'] = df['zip code'].astype(str) 
zip_codes['ZIP'] = zip_codes['ZIP'].astype(str)



# In[19]:


# merge the dataframes
df = pd.merge(df, zip_codes, left_on='zip code', right_on='ZIP', how='inner')

# In[20]:


# calculate the time difference between the release date and the timestamp
df['release_date'] = pd.to_datetime(df['release_date'])
df['days_after_release'] = df['timestamp'] - df['release_date']
df['days_after_release'] = df['days_after_release'].dt.days



# In[21]:


# add a column for the year of the release
df['release-year'] = df['release_date'].dt.year



# In[ ]:





# In[22]:


df['rating'] = pd.to_numeric(df['rating'], downcast='unsigned')
df['age'] = pd.to_numeric(df['age'], downcast='unsigned')





# ## Data Visualisation

# In[25]:
def load_vis():

# make a countplot by occupation
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='occupation', data=df, palette='viridis')

    # make labels
    plt.xticks(rotation=72)
    plt.title('Countplot by Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('Count')



    # In[26]:


    # make a countplot of the gender
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gender', data=df, palette='viridis')

    # make
    plt.title('Countplot by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')

    # Show the plot



    # In[27]:


    # show the distribution of people by age and gender
    # Define age groups
    bins = [0, 18, 25, 35, 45, 50, 56, 100]
    labels = ['<18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Plot the count of users by age group and gender
    plt.figure(figsize=(14, 8))
    sns.countplot(x='age_group', hue='gender', data=df, palette='viridis')
    plt.title('User Demographics by Age Group and Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Count')


    # print the number of people in each age group
    age_group_counts = df['age_group'].value_counts().sort_index()



    # In[28]:


    df.info()


# In[29]:


    # plot the mean rating for eeach genre
    genre_columns = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    # find mean rating in each genre
    genre_ratings = {}
    for genre in genre_columns:
        genre_ratings[genre] = df.loc[df[genre] == 1, 'rating'].mean()

    # convert the dictionary to a DataFrame for plotting
    genre_ratings_df = pd.DataFrame(list(genre_ratings.items()), columns=['Genre', 'Mean Rating'])

    # plot the mean ratings for each genre
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Genre', y='Mean Rating', data=genre_ratings_df, palette='viridis')
    plt.title('Mean Ratings for Each Genre')
    plt.xlabel('Genre')
    plt.ylabel('Mean Rating')
    plt.xticks(rotation=45)



# ## Hybrid Filtering Model

# In[30]:


 




# In[31]:


# create a user-level dataframe with demographic information
user_demo_df = df[['user_id', 'age', 'gender', 'occupation', 'STATE']].drop_duplicates('user_id')

# make gender a numeric value
user_demo_df['gender'] = user_demo_df['gender'].map({'M': 1, 'F': 0})

# encode occupation a categorical value
occ_encoder = LabelEncoder()
user_demo_df['occupation'] = occ_encoder.fit_transform(user_demo_df['occupation'])

# encode STATE as numeric (location).
state_encoder = LabelEncoder()
user_demo_df['STATE'] = state_encoder.fit_transform(user_demo_df['STATE'])

# normalise the features
scaler = MinMaxScaler()
user_demo_df[['age', 'gender', 'occupation', 'STATE']] = scaler.fit_transform(
    user_demo_df[['age', 'gender', 'occupation', 'STATE']]
)

# Rating-based similarity
ratings_df = df[['user_id', 'item_id', 'rating']]
scaler.fit_transform(ratings_df[['rating']])



# In[32]:


# Rating-based similarity
ratings_df = df[['user_id', 'item_id', 'rating']]
scaler.fit_transform(ratings_df[['rating']])


# ### Compute demographic similarity

# In[33]:


# compute demographic similarity using user_demo_df

# first set 'user_id' as the index for clarity
user_demo_df = user_demo_df.set_index('user_id')

# feed just the numeric columns (age, gender, occupation, STATE)
demographic_features = user_demo_df[['age', 'gender', 'occupation', 'STATE']].values

# cosine similarity across all users
demo_sim = cosine_similarity(demographic_features, demographic_features)

# Convert to DataFrame with user_ids as index/columns
demo_sim_df = pd.DataFrame(
    demo_sim, 
    index=user_demo_df.index, 
    columns=user_demo_df.index
)




# In[34]:


from scipy.sparse.linalg import svds
# We assume 'ratings_df' has columns: [user_id, item_id, rating]
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Get unique user_ids and item_ids (for building factor matrices)
unique_users = ratings_df['user_id'].unique()
unique_items = ratings_df['item_id'].unique()
num_users = len(unique_users)
num_items = len(unique_items)

# Map user_id -> row index, item_id -> column index
user_to_index = {u: idx for idx, u in enumerate(unique_users)}
item_to_index = {i: idx for idx, i in enumerate(unique_items)}

def train_mf_sgd(
    train_data, user_to_index, item_to_index,
    num_users, num_items,
    k=10,         # latent factors
    alpha=0.007,  # learning rate
    reg=0.1,      # regularization
    epochs=10
):
    """
    Train an MF model using SGD and return user_factors, item_factors.
    Prints epoch-wise RMSE on the training data.
    """
    rng = np.random.default_rng(seed=42)
    
    # Randomly initialize latent factors
    user_factors = rng.normal(scale=0.1, size=(num_users, k))  
    item_factors = rng.normal(scale=0.1, size=(num_items, k))  
    
    # Convert train_data into list of (user_idx, item_idx, rating)
    train_tuples = []
    for row in train_data.itertuples():
        u_idx = user_to_index[row.user_id]
        i_idx = item_to_index[row.item_id]
        rating = row.rating
        train_tuples.append((u_idx, i_idx, rating))

    # list to hold the RMSE values
    rmse_values = []
    
    # SGD training
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_tuples)
        
        for (u_idx, i_idx, r) in train_tuples:
            pred = np.dot(user_factors[u_idx], item_factors[i_idx])
            err = r - pred  # error = actual - predicted
            
            uf = user_factors[u_idx]  # copy reference
            itf = item_factors[i_idx]
            
            # Update user_factors
            user_factors[u_idx] += alpha * (err * itf - reg * uf)
            # Update item_factors
            item_factors[i_idx] += alpha * (err * uf - reg * itf)
        
        # Compute train RMSE at end of epoch
        train_preds = []
        train_actuals = []
        for (u_idx, i_idx, r) in train_tuples:
            p = np.dot(user_factors[u_idx], item_factors[i_idx])
            train_preds.append(p)
            train_actuals.append(r)
        train_rmse = np.sqrt(mean_squared_error(train_actuals, train_preds))
        #print(f"Epoch {epoch}/{epochs} - Train RMSE: {train_rmse:.4f}")
        rmse_values.append(train_rmse)

    return user_factors, item_factors, rmse_values

# Train the model
user_factors, item_factors, rmse_values = train_mf_sgd(
    train_data=train_data,
    user_to_index=user_to_index,
    item_to_index=item_to_index,
    num_users=num_users,
    num_items=num_items,
    k=10,        # latent factors
    alpha=0.007, # learning rate
    reg=0.1,     # regularization
    epochs=10    # number of epochs
)


# plot the RMSE values
# Suppose rmse_values is the list of RMSE values returned from the training function.
epochs = range(1, len(rmse_values) + 1)
def epoc_plot(rmse_values):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, rmse_values, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training RMSE per Epoch')
    plt.grid(True)





# In[35]:


# Basic function to get MF rating prediction for (user, item)
def predict_rating_mf(user_id, item_id, user_factors, item_factors, 
                      user_to_index, item_to_index):
    if user_id not in user_to_index or item_id not in item_to_index:
        return np.nan
    u_idx = user_to_index[user_id]
    i_idx = item_to_index[item_id]
    return np.dot(user_factors[u_idx], item_factors[i_idx])
# Example usage
user_id = 1
item_id = 1
predicted_rating = predict_rating_mf(user_id, item_id, user_factors, item_factors, user_to_index, item_to_index)


def predict_rating_demo(user_id, item_id, train_utility, demo_sim_df, k=10):
    """
    Predict the rating for (user_id, item_id) using top-k neighbors 
    in the demographic similarity matrix 'demo_sim_df'.
    'train_utility' is a pivot table [user_id x item_id] of known ratings.
    """
    # If the item doesn't exist in train_utility columns, no prediction
    if item_id not in train_utility.columns:
        return np.nan
    
    # Users who rated this item
    item_ratings = train_utility[item_id]
    valid_users = item_ratings.dropna().index  # user_ids who rated 'item_id'
    
    if user_id not in demo_sim_df.index:
        return np.nan
    
    # Get similarities to user_id
    similarities = demo_sim_df.loc[user_id]
    # Filter to users who rated the item, then sort desc
    neighbors = similarities.loc[valid_users].sort_values(ascending=False).head(k)
    
    num = 0
    den = 0
    for neighbor_id, sim_val in neighbors.items():
        neighbor_rating = item_ratings.loc[neighbor_id]
        num += sim_val * neighbor_rating
        den += abs(sim_val)
    if den == 0:
        return np.nan
    return num / den

def predict_rating_hybrid_mf_demo(
    user_id, item_id,
    alpha,  # e.g. 0.7
    user_factors, item_factors, 
    user_to_index, item_to_index,
    train_utility, demo_sim_df, 
    k_demo=10
):
    """
    Hybrid rating = alpha * (MF rating) + (1 - alpha) * (Demographic rating).
    """
    # 1. MF prediction
    mf_pred = predict_rating_mf(
        user_id, item_id, 
        user_factors, item_factors, 
        user_to_index, item_to_index
    )
    
    # 2. Demographic-based prediction
    demo_pred = predict_rating_demo(
        user_id, item_id, 
        train_utility, demo_sim_df, 
        k=k_demo
    )
    
    # fallback logic if one is NaN
    if np.isnan(mf_pred) and np.isnan(demo_pred):
        return np.nan
    if np.isnan(mf_pred):
        return demo_pred
    if np.isnan(demo_pred):
        return mf_pred
    
    # Weighted combination
    return alpha * mf_pred + (1 - alpha) * demo_pred




# In[36]:


# Build a pivot from train_data for the demographic CF
train_utility = train_data.pivot_table(values='rating', index='user_id', columns='item_id')


actuals = []
preds = []
alpha = 0.8

for row in test_data.itertuples():
    u = row.user_id
    i = row.item_id
    actual_rating = row.rating
    
    pred = predict_rating_hybrid_mf_demo(
        user_id=u,
        item_id=i,
        alpha=alpha,
        user_factors=user_factors,
        item_factors=item_factors,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        train_utility=train_utility,
        demo_sim_df=demo_sim_df,
        k_demo=10
    )
    
    if not np.isnan(pred):
        actuals.append(actual_rating)
        preds.append(pred)

test_rmse = np.sqrt(mean_squared_error(actuals, preds))


# ### Novelty Score

# In[37]:


import math
import pandas as pd

# total number of user-item ratings in the dataset
item_counts = ratings_df.groupby('item_id')['user_id'].count()

# Convert to a dictionary for faster access
item_pop = item_counts.to_dict()

# total number of ratings in the dataset
total_ratings = item_counts.sum()

def novelty_score(recommended_items, item_pop, total_ratings):
    """
    Compute the average negative log-base-2 popularity for a list of items.
    Items with lower popularity (rarer items) have higher novelty.
    
    recommended_items : list of item_ids
    item_pop          : dict { item_id -> number of ratings }
    total_ratings     : int   total number of (user, item) ratings in the dataset
    
    Returns:
      A float representing the average novelty for the recommended items.
      If none have a known popularity, returns 0.
    """
    import math
    
    if not recommended_items:
        return 0.0
    
    novelty_sum = 0.0
    valid_count = 0
    
    for item_id in recommended_items:
        count_i = item_pop.get(item_id, 0)
        if count_i > 0:
            popularity_fraction = count_i / total_ratings
            # novelty = -log2(popularity fraction)
            novelty_sum += -math.log2(popularity_fraction)
            valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    return novelty_sum / valid_count


def recommend_hybrid(
    user_id,
    alpha,
    user_factors,
    item_factors,
    user_to_index,
    item_to_index,
    train_utility,
    demo_sim_df,
    df,
    item_pop,
    total_ratings,
    k_demo=10,
    top_n=5
):
    """
    Generate top-N movie recommendations for a given user using a hybrid MF + demographic approach,
    then compute a novelty score for those top-N items.
    """
    # 1) Identify items user already rated (skip them)
    if user_id in train_utility.index:
        rated_items = train_utility.loc[user_id].dropna().index
    else:
        rated_items = []

    # 2) All known items from your item_to_index dict
    all_items = list(item_to_index.keys())

    predictions = {}

    # 3) Predict rating for each not-yet-rated item
    for item_id in all_items:
        if item_id not in rated_items:
            pred = predict_rating_hybrid_mf_demo(
                user_id=user_id,
                item_id=item_id,
                alpha=alpha,
                user_factors=user_factors,
                item_factors=item_factors,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                train_utility=train_utility,
                demo_sim_df=demo_sim_df,
                k_demo=k_demo
            )
            if not np.isnan(pred):
                predictions[item_id] = pred

    # 4) Sort by predicted rating descending
    if not predictions:
        return [], 0.0  # no recommendations => no novelty
    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_recs = sorted_items[:top_n]

    # 5) Look up titles & store final recs
    rec_items = [item_id for (item_id, score) in top_recs]
    rec_list = []
    for (item_id, score) in top_recs:
        movie_title = df.loc[df['item_id'] == item_id, 'movie_title'].iloc[0] \
                      if not df.loc[df['item_id'] == item_id].empty else "Unknown Title"
        rec_list.append((item_id, score, movie_title))
    
    # 6) Compute novelty for these top-N items
    avg_novelty = novelty_score(rec_items, item_pop, total_ratings)
    
    return rec_list, avg_novelty

# Example usage
recommended_items, novelty = recommend_hybrid(
    user_id=10,
    alpha=0.7,
    user_factors=user_factors,
    item_factors=item_factors,
    user_to_index=user_to_index,
    item_to_index=item_to_index,
    train_utility=train_utility,
    demo_sim_df=demo_sim_df,
    df=df,  # your main DataFrame with movie titles
    item_pop=item_pop,  # dict { item_id -> # of ratings }
    total_ratings=total_ratings,  # total rating count
    k_demo=10,
    top_n=5
)




# ## Recommender System 2 

# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def ratings_vis(df):
# Set a whitegrid style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # 1. Normalize the 'rating' column in your df
    scaler = MinMaxScaler()
    df['rating'] = scaler.fit_transform(df[['rating']])

    # 2. Visualize the distribution of normalized ratings
    plt.subplot(1, 2, 1)
    sns.histplot(df['rating'], bins=50, color='lightgreen')
    plt.title('Distribution of Ratings (After Normalization)', fontsize=16)
    plt.xlabel('Normalized Rating', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    plt.tight_layout()
    plt.show()

# 3. Convert user_id and movie_id to categorical codes
user_ids = df['user_id'].astype('category').cat.codes.values
item_ids = df['movie_id'].astype('category').cat.codes.values

# 4. Prepare the input (X) and target (y)
X = np.stack([user_ids, item_ids], axis=1)
y = df['rating'].values

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MovieDataset(Dataset):
    def __init__(self, X, y):
        """
        X should be a 2D NumPy array with:
           X[:, 0] = user_id (as integer index)
           X[:, 1] = item_id (as integer index)
        y should be a 1D NumPy array of ratings (floats).
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (user_id, item_id, rating)
        return self.X[idx, 0], self.X[idx, 1], self.y[idx]

# Create train and test datasets (assumes X_train, y_train, X_test, y_test exist)
train_dataset = MovieDataset(X_train, y_train)
test_dataset = MovieDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#print(f"Size of the training dataset: {len(train_dataset)}")
#print(f"Size of the testing dataset: {len(test_dataset)}")

# Visualize a sample batch of training data




# In[41]:


import torch
import torch.nn as nn

# 1. Compute number of unique users/items
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()

class DMFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, user_layers_sizes, item_layers_sizes):
        super(DMFModel, self).__init__()

        # User embedding + deep layers
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.user_layers = nn.ModuleList()
        for in_size, out_size in zip([embedding_size] + user_layers_sizes[:-1], user_layers_sizes):
            self.user_layers.append(nn.Linear(in_size, out_size))
            self.user_layers.append(nn.ReLU())

        # Item embedding + deep layers
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        self.item_layers = nn.ModuleList()
        for in_size, out_size in zip([embedding_size] + item_layers_sizes[:-1], item_layers_sizes):
            self.item_layers.append(nn.Linear(in_size, out_size))
            self.item_layers.append(nn.ReLU())

        # Final prediction layer
        self.output = nn.Linear(user_layers_sizes[-1] + item_layers_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        # User part
        user_embedded = self.user_embedding(user_ids)
        for layer in self.user_layers:
            user_embedded = layer(user_embedded)

        # Item part
        item_embedded = self.item_embedding(item_ids)
        for layer in self.item_layers:
            item_embedded = layer(item_embedded)

        # Concatenate and output
        concatenated = torch.cat([user_embedded, item_embedded], dim=-1)
        x = self.sigmoid(self.output(concatenated))
        return x

# 2. Define hyperparameters
embedding_size = 80
user_layers_sizes = [50, 50]
item_layers_sizes = [50, 50]

# 3. Create an instance of the DMF model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DMFModel(n_users, n_items, embedding_size, user_layers_sizes, item_layers_sizes).to(device)

# 4. Display the model architecture




# In[42]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

# Example hyperparameters, model, and data
# ----------------------------------------
# model = DMFModel(...)  # your PyTorch model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# train_loader, test_loader = ...  # your DataLoaders
# scaler = ...  # the MinMaxScaler used to scale df['rating']
# X_test, y_test = ...  # your NumPy arrays for test data (unscaled or scaled as needed)
# ----------------------------------------

# Initialize the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
patience = 3
best_loss = np.inf
epochs_no_improve = 0
criterion = nn.MSELoss()

# Lists to store training and testing losses for visualization
train_losses = []
test_losses = []

# Training loop
for epoch in range(10):
    # Training phase
    model.train()
    train_loss = 0
    for user_ids, item_ids, ratings in train_loader:
        # Move data to the same device as the model
        user_ids = user_ids.to(device).long()
        item_ids = item_ids.to(device).long()
        ratings = ratings.to(device).float()

        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(user_ids, item_ids).squeeze()
        
        # Compute loss
        loss = criterion(predictions, ratings)
        
        # Backprop + optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Evaluation phase
    model.eval()
    y_pred = []
    test_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device).long()
            item_ids = item_ids.to(device).long()
            ratings = ratings.to(device).float()

            predictions = model(user_ids, item_ids).squeeze()
            y_pred.extend(predictions.cpu().numpy())
            
            # Calculate batch loss
            loss = criterion(predictions, ratings)
            test_loss += loss.item()

    # Average out losses
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    
    #print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Early stopping check
    if test_loss < best_loss:
        best_loss = test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        break

# Visualize training & testing losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# print the lowest test loss
# print(f"Lowest Test Loss: {min(test_losses):.4f}")


# Invert scaling of predictions and compute RMSE
# y_pred is predicted ratings in scaled form
# y_test is your ground truth ratings (scaled); ensure shapes match

y_pred_rescaled = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))



# In[ ]:


import torch
import pandas as pd
import numpy as np

def recommend_items(
    model,
    user_id,
    user_id_mapping,
    item_id_mapping,
    df,
    n_recommendations
):
    """
    Recommend items for a given user_id using the trained model.
    Returns a dataframe with [movie_title, predicted_rating, novelty].
    """
    # 1) Ensure the user ID is in the mapping
    if user_id not in user_id_mapping:
        print(f"User ID {user_id} not found in user_id_mapping.")
        return pd.DataFrame(), []

    # 2) Find which items the user has already rated
    rated_items = df.loc[df['user_id'] == user_id, 'item_id'].unique()
    # Build a list of "unseen" items
    unseen_items = [iid for iid in item_id_mapping if iid not in rated_items]

    # 3) Convert user_id to the model's embedding index
    user_index = user_id_mapping[user_id]
    user_index_tensor = torch.tensor([user_index], device=model.user_embedding.weight.device)

    # 4) Predict ratings for all unseen items
    predicted_ratings = []
    for item_id in unseen_items:
        item_index = item_id_mapping[item_id]
        item_index_tensor = torch.tensor([item_index], device=model.item_embedding.weight.device)
        
        # Forward pass through model
        pred_rating = model(user_index_tensor, item_index_tensor).item()
        predicted_ratings.append((item_id, pred_rating))
    
    # 5) Sort items by predicted rating in descending order, pick top-N
    top_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:n_recommendations]
    top_item_ids = [t[0] for t in top_items]
    
    # 6) Merge metadata from df
    #    df likely has multiple rows per (user_id, item_id),
    #    so let's take item-level metadata by dropping duplicates on item_id
    item_metadata = df.drop_duplicates(subset='item_id').copy()
    
    # Keep only rows for top_item_ids
    recommended = item_metadata.loc[item_metadata['item_id'].isin(top_item_ids)].copy()
    
    # Create a mapping from item_id -> predicted rating
    id_to_pred = dict(top_items)
    recommended['predicted_rating'] = recommended['item_id'].map(id_to_pred)
    
    # 7) Compute novelty for each recommended item.
    #    We'll define novelty as: novelty = -log2( (number of ratings for the item) / (total ratings) )
    counts = df['item_id'].value_counts()  # popularity of each item
    total_ratings = counts.sum()
    # Use a small value if an item has 0 ratings (shouldn't happen in practice)
    recommended['novelty'] = recommended['item_id'].apply(
        lambda x: -np.log2(counts.get(x, 1e-10) / total_ratings)
    )
    
    # 8) Return only movie_title, predicted_rating, and novelty
    columns_to_return = ['movie_title', 'predicted_rating', 'novelty']
    recommended = recommended[columns_to_return].drop_duplicates().copy()
    
    # Sort by predicted rating descending
    recommended.sort_values(by='predicted_rating', ascending=False, inplace=True)

    # print average novelty
    avg_novelty = recommended['novelty'].mean()
    #print(f"Average Novelty for User {user_id}: {avg_novelty:.4f}")

    
    
    return recommended, top_item_ids
#create mappings for user and item IDs
user_id_mapping = {uid: i for i, uid in enumerate(df['user_id'].unique())}
item_id_mapping = {iid: i for i, iid in enumerate(df['item_id'].unique())}
# Example usage

recommended_items_df, top_item_ids = recommend_items(
    model,
    user_id,
    user_id_mapping,
    item_id_mapping,
    df,
    n_recommendations=5
)

import argparse
# CLI
def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument('--user_id', type=int, required=True, help='User ID for which to recommend movies')
    parser.add_argument('--num_recs', type=int, default=10, help='Number of movie recommendations to generate')
    parser.add_argument('--eval' , action='store_true', help='Evaluate the model')
    parser.add_argument('--basic', action='store_true', help='Use basic recommendation system')
    parser.add_argument('--advanced', action='store_true', help='Use advanced recommendation system')
    args = parser.parse_args()

    # if evaluation flag is set, evaluate the model
    if args.eval:
        print("The metrics for the basic model are:")
        print(f"RMSE: {test_rmse:.4f}")


        print("The metrics for the advanced model are:")
        print(f"RMSE: {min(test_losses):.4f}\n")

    if args.basic:
        if args.user_id not in user_id_mapping:
            print(f"User ID {args.user_id} not found in user_id_mapping.")
            return
        elif args.user_id is None:
            print("Please enter a user ID")
        else:
            print("Running advanced recommendation system for user ID:", args.user_id)
            # Call your evaluation function here
            recommended_items, novelty = recommend_hybrid(
                user_id=args.user_id,
                alpha=0.7,
                user_factors=user_factors,
                item_factors=item_factors,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                train_utility=train_utility,
                demo_sim_df=demo_sim_df,
                df=df,  
                item_pop=item_pop,  
                total_ratings=total_ratings,  
                k_demo=10,
                top_n=args.num_recs
            )
            print("Recommended Items:")
            for item_id, score, title in recommended_items:
                print(f"Item ID: {item_id}, Score: {score:.4f}, Title: {title}")
            print(f"Average Novelty Score: {novelty:.4f}")

    if args.advanced:
        if args.user_id not in user_id_mapping:
            print(f"User ID {args.user_id} not found in user_id_mapping.")
            return
        elif args.user_id is None:
            print("Please enter a user ID")
        else:
            print("Running advanced recommendation system for user ID:", args.user_id)
            user_id = 1
            recommended_items_df = recommend_items(
                model,
                user_id,
                user_id_mapping = user_id_mapping,
                item_id_mapping = item_id_mapping,
                df=df,
                n_recommendations=args.num_recs
            )
            # Display the recommended items
            print("Recommended Items:")
            print(recommended_items_df)
            

if __name__ == "__main__":
    main()

# Code Used for visualisation and preparation
# [1] N. Bousabat, "Movie Recommendation System Project," Kaggle, Jan. 2025. [Online]. Available: https://www.kaggle.com/code/nizarbousabat/movie-recommendation-system-project#Clean-the-data


