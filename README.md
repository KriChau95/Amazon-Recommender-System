# Amazon-Recommender-System
A Python Data Science project that leverages NumPy, Pandas, Matplotlib to implement an item-item collaborative filtering algorithm on a 3,000+ item dataset.
This is a group project worked on by myself, Kelvin Bian, and Kishan Patel.

## Libraries used
* `numpy`
* `matplotlib`
* `pandas`
* `json`
* `sklearn`
* `scipy`
* `warnings`
* `gzip`
* `surprise`
* `tqdm`

## Dataset
The dataset we worked with is the Amazon Luxury Beauty Dataset, which is a collection of 3,000+ items such as perfume
cologne, makeup, foundation, brush, and skincare products.

## Justification for Item-Item Collaborative Filtering

In e-commerce, items are more stable than users, whose preferences change frequently. This makes items a reliable basis for recommendations.

### Process:
1. **Determine Item Signatures**: Based on user ratings in the user-item matrix.  
2. **Find Similar Items**: Identify items rated by the user that are similar to the target item.  
3. **Predict Ratings**: Calculate the weighted sum of ratings for similar items to generate a recommendation.

This method leverages item stability to provide consistent and personalized recommendations.

## Step 1: Data Preprocessing

The dataset is filtered to retain only relevant metrics:  
- **`asin`**: Product ID  
- **`reviewerID`**: User ID  
- **`overall`**: Rating  

The data is grouped by `reviewerID` to analyze user-specific interactions. A random seed is set to ensure the reproducibility of results across different runs of the model. This guarantees consistent outputs for evaluation and comparison. 

Split data into training data set (80%) and testing data set (20%).

Handle duplicate (user, item) pairs by averaging all of a user's reviews for a specific item.

## Step 2: Item-Item Collaborative Filtering

1. Compute an Item-Item similarity matrix across all items using cosine similarity as the similarity metric.
2. Make predictions on test set - for each item in test set, choose 5 most similar items and compute weighted average of corresponding similar items

## Step 3: Predicition Evaluation

Use accuracy metrics such as RMSE and MAE to gauge model accuracy.

## Step 4: Item Recommendations

Based on the information derived from Item-Item CF, recommend 10 items to each user.

## Step 5: Recommendation Evaluation

Use metrics such as Precision, Recall, and NDCG to assess quality of recommendations.

## Step 6: Further Exploration

Compare this approach with other approaches:
1. Item-Item Collaborative Filtering with a Baseline estimate derived from global mean, user deviation, and item deviation.
2. Content based filtering, incorporating TF-IDF.
3. SVD - Singular Value Decomposition to construct matrices that encapsulates the patterns of the data, and tunes the values to fit the dataset.









