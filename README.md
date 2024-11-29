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





