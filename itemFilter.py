import pandas as pd
import gzip
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Step 1: Load and Parse the Data
def parse(path):
    with gzip.open(path, 'rb') as g:
        for line in g:
            yield json.loads(line)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Load data
df = getDF('Luxury_Beauty_5.json.gz')

# Step 2: Data Preparation
# Keep only the relevant columns
df = df[['reviewerID', 'asin', 'overall']]

# Handle duplicates in train_df by averaging duplicate entries
df = df.groupby(['reviewerID', 'asin'], as_index=False).mean()

# Group by each user (reviewerID)
user_groups = df.groupby('reviewerID')

# Set random seed for reproducibility
np.random.seed(42) 

# Split the data into training and testing sets
train_data = []
test_data = []

for _, group in user_groups:
    ratings = group.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(ratings) * 0.8)  # 80% train, 20% test
    if len(ratings[:split_idx]) > 0 and len(ratings[split_idx:]) > 0:
        train_data.append(ratings[:split_idx])
        test_data.append(ratings[split_idx:])

# Concatenate all user data to form the final training and testing datasets
train_df = pd.concat(train_data).reset_index(drop=True)
test_df = pd.concat(test_data).reset_index(drop=True)
print("\nSize of test data set:",len(test_df))

# Step 3: Implementing Item-Item Collaborative Filtering
# Create a pivot table for training data (users x items matrix)
train_pivot = train_df.pivot(index='reviewerID', columns='asin', values='overall').fillna(0)

# Calculate item-item similarity matrix using cosine similarity

item_similarity = cosine_similarity(train_pivot.T)
item_similarity_df = pd.DataFrame(item_similarity, index=train_pivot.columns, columns=train_pivot.columns)

# Function to predict rating based on item similarity
def predict_rating(user, item, k=5):
    if item not in item_similarity_df.index:
        return np.nan  # If the item is unknown, we cannot make a prediction
    
    # Find k most similar items that the user has rated
    similar_items = item_similarity_df[item].sort_values(ascending=False)[1:k+1]
    rated_items = train_pivot.loc[user].loc[similar_items.index]
    rated_items = rated_items[rated_items > 0]  # Keep only items that have been rated
    
    if rated_items.empty:
        return np.nan  # If there are no similar rated items, return NaN
    
    # Compute the weighted sum of ratings
    similarity_sum = similar_items.loc[rated_items.index].sum()
    if similarity_sum == 0:
        return np.nan  # Avoid division by zero
    
    return np.dot(rated_items, similar_items.loc[rated_items.index]) / similarity_sum

# Make predictions for the testing set
test_df['predicted_rating'] = test_df.apply(lambda x: predict_rating(x['reviewerID'], x['asin']), axis=1)

# Drop any rows where prediction is NaN (due to cold start problem or missing items)
test_df = test_df.dropna(subset=['predicted_rating'])
print("Number of predictions made", len(test_df))

# Step 4: Calculate MAE and RMSE
def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

mae = calculate_mae(test_df['overall'], test_df['predicted_rating'])
rmse = calculate_rmse(test_df['overall'], test_df['predicted_rating'])

print("Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 5: make recommendations
def precompute_recommendations(train_pivot, item_similarity_df, k=10):
    # Convert training pivot to a sparse matrix
    train_sparse = csr_matrix(train_pivot.values)
    
    # Convert item similarity to a sparse matrix
    item_similarity_sparse = csr_matrix(item_similarity_df.values)
    
    # Predict ratings for all users and all items
    all_predictions = train_sparse.dot(item_similarity_sparse).toarray()
    
    # Mask out already-rated items
    rated_mask = (train_sparse > 0).toarray()  # Mask of rated items
    all_predictions[rated_mask] = -np.inf  # Ensure rated items are not recommended
    
    # Generate top-k recommendations for each user
    recommendations = {}
    for user_idx, user_id in enumerate(train_pivot.index):
        user_predictions = all_predictions[user_idx]
        # Ensure k does not exceed the number of valid recommendations
        valid_items_mask = user_predictions != -np.inf  # Identify items with valid predictions
        available_items = valid_items_mask.sum()
        top_k = min(k, available_items)
        
        if top_k > 0:
            top_k_items_idx = np.argpartition(user_predictions, -top_k)[-top_k:]  # Get indices of top-k items
            top_k_items_idx = top_k_items_idx[np.argsort(user_predictions[top_k_items_idx])[::-1]]  # Sort indices
            top_k_items = train_pivot.columns[top_k_items_idx].tolist()  # Map indices back to item IDs
            recommendations[user_id] = top_k_items
        else:
            recommendations[user_id] = []  # No recommendations available for this user
    
    return recommendations

# Precompute recommendations
recommendations = precompute_recommendations(train_pivot, item_similarity_df, k=10)

# Output recommendations to a txt file called recommendations.txt
with open("recommendations.txt", "w") as file:
    # Iterate over the recommendations and write each to the file
    for user, rec_list in recommendations.items():
        file.write(f"Recommendations for user {user}: {rec_list}\n")

# Step 6: Evaluate Recommendations
def evaluate_recommendations(recommendations, test_df, k=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    # Get the list of users in the test set
    test_users = test_df['reviewerID'].unique()

    for user, rec_list in recommendations.items():
        # Only evaluate users who are in the test set
        if user not in test_users:
            continue
        
        # Get the actual items the user purchased in the test set
        test_items = test_df[test_df['reviewerID'] == user]['asin'].values
        
        # Calculate Precision
        hits = len(set(rec_list) & set(test_items))
        precision = hits / k if k > 0 else 0
        precision_list.append(precision)
        
        # Calculate Recall
        recall = hits / len(test_items) if len(test_items) > 0 else 0
        recall_list.append(recall)
        
        # Calculate NDCG
        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(rec_list) if item in test_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(test_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    # Compute average metrics
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)
    
    # Compute F-measure
    f_measure = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f_measure, avg_ndcg

# Evaluate recommendations
avg_precision, avg_recall, f_measure, avg_ndcg = evaluate_recommendations(recommendations, test_df)

# Print evaluation metrics
print("\nRecommendation Metrics:")
print(f"Precision: {avg_precision}")
print(f"Recall: {avg_recall}")
print(f"F-measure: {f_measure}")
print(f"NDCG: {avg_ndcg}")

# Plot residual graph

# Calculate residuals
test_df['residuals'] = test_df['overall'] - test_df['predicted_rating']

# Plot histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(test_df['residuals'], bins=30, edgecolor='black', alpha=0.7, align='mid')
plt.xlabel('Prediction Error (Residuals)')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.show()