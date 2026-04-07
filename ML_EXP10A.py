print("JANASREE 24BAD040")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
import seaborn as sns

ratings = pd.read_csv(r"C:\Users\janas\Downloads\ratings.csv")
movies = pd.read_csv(r"C:\Users\janas\Downloads\movies.csv")

ratings = ratings[['userId', 'movieId', 'rating']]
movies = movies[['movieId', 'title']]

df = pd.merge(ratings, movies, on='movieId')

print(df.head())

user_item_matrix = df.pivot_table(index='userId',
                                  columns='title',
                                  values='rating')

user_item_matrix = user_item_matrix.fillna(0)

matrix = user_item_matrix.values

user_mean = np.mean(matrix, axis=1).reshape(-1, 1)
matrix_norm = matrix - user_mean

k = 20

svd = TruncatedSVD(n_components=k, random_state=42)
matrix_reduced = svd.fit_transform(matrix_norm)
matrix_reconstructed = np.dot(matrix_reduced, svd.components_) + user_mean

predicted_df = pd.DataFrame(matrix_reconstructed,
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)

mask = matrix != 0
actual = matrix[mask]
predicted = matrix_reconstructed[mask]

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(8, 6))
sns.heatmap(user_item_matrix.iloc[:10, :10], cmap='YlGnBu')
plt.title("Original User-Item Matrix (Sample)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(predicted_df.iloc[:10, :10], cmap='YlGnBu')
plt.title("Reconstructed Matrix using SVD")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

k_values = [5, 10, 20, 30, 40]
rmse_values = []

for k in k_values:
    svd = TruncatedSVD(n_components=k, random_state=42)
    reduced = svd.fit_transform(matrix_norm)
    reconstructed = np.dot(reduced, svd.components_) + user_mean
    pred = reconstructed[mask]
    rmse_val = np.sqrt(mean_squared_error(actual, pred))
    rmse_values.append(rmse_val)

plt.figure(figsize=(8, 6))
plt.plot(k_values, rmse_values, marker='o', label='RMSE')
plt.title("Error vs Number of Latent Factors")
plt.xlabel("Number of Latent Factors (k)")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

user_id = 1

if user_id in predicted_df.index:
    user_ratings = predicted_df.loc[user_id]
    original_rated = user_item_matrix.loc[user_id]
    recommendations = user_ratings[original_rated == 0].sort_values(ascending=False)

    print("\nTop 10 Recommended Movies for User", user_id)
    print(recommendations.head(10))
else:
    print("User ID not found in dataset")
