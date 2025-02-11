# Load the libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

# Load the model
# reg = load(open('./model/regressor.pkl','rb'))
movies_dict = load(open('./model/movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
# similarity = load(open('./model/similarity.pkl','rb'))



# amazon_ratings = pd.read_csv('./input/amazon-ratings/ratings_Beauty.csv')
# amazon_ratings = amazon_ratings.dropna()
# amazon_ratings.head()

response = requests.get('http://localhost:3000/api/rating')
data = response.json()
amazon_ratings = pd.DataFrame(data)
amazon_ratings = amazon_ratings.dropna()
print(amazon_ratings)

movies_return = []
for i in movies["title"]:
    movies_return.append(i)

# print(len(movies_return))

def recommend_product_by_rating():
    popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    print(most_popular.head(10))


# Initialize an instance of FastAPI
app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the default route 
@app.get("/")
def root():
    return {"message": data}

@app.get("/recommend_product_by_top_rating")
def root():
    popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    top_products = most_popular.head(10).reset_index().rename(columns={'Rating': 'Rating'}).to_dict(orient='records')
    return {"top_products": top_products}

@app.post("/recommend_product_by_similar_rating")
def root(productId: str):
    amazon_ratings1 = amazon_ratings.head(10000)
    # print(amazon_ratings1)
    ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
    ratings_utility_matrix.head()
    print(ratings_utility_matrix)
    X = ratings_utility_matrix.T
    X.head()
    # print(X.shape)
    X1 = X
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    decomposed_matrix.shape
    correlation_matrix = np.corrcoef(decomposed_matrix)
    correlation_matrix.shape
    # X.index[99]
    # i = "6117036094"
    i = productId

    product_names = list(X.index)
    product_ID = product_names.index(i)
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    Recommend = list(X.index[correlation_product_ID > 0.90])

    # Removes the item already bought by the customer
    Recommend.remove(i) 

    Recommend[0:9]
    return {"top_products": Recommend[0:9],"product_id": product_ID}

@app.post("/recommend_product_by_description")
def root(keyword: str):
    cluster_terms = []
    res = requests.get('http://localhost:3000/api/description')
    data_descriptions = res.json()

    # product_descriptions = pd.read_csv('./input/home-depot-product-search-relevance/product_descriptions.csv')
    product_descriptions = pd.DataFrame(data_descriptions)
    product_descriptions.shape
    
    product_descriptions = product_descriptions.dropna()
    product_descriptions.shape
    product_descriptions.head()

    product_descriptions1 = product_descriptions.head(500)
    # product_descriptions1.iloc[:,1]

    product_descriptions1["product_description"].head(10)

    with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
        stop_words_vietnamese = [word.strip() for word in f.readlines()]
    print(stop_words_vietnamese)


    vectorizer = TfidfVectorizer(stop_words=stop_words_vietnamese)
    X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
    X1
    
    X=X1

    kmeans = KMeans(n_clusters = 10, init = 'k-means++')
    y_kmeans = kmeans.fit_predict(X)
    # plt.plot(y_kmeans, ".")
    # plt.show()

    def print_cluster(i):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            # cluster_terms.append(terms[ind])
            print(' %s' % terms[ind]),
        print

    def print_cluster_recommend(i):
        print("Recommend Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            cluster_terms.append(terms[ind])
            print(' %s' % terms[ind]),
        print

    true_k = 10

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X1)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print_cluster(i)
        
    cluster_terms = []

    def show_recommendations(product):
        print("Cluster Recommend ID:")
        Y = vectorizer.transform([product])
        prediction = model.predict(Y)
        # print(prediction)
        print_cluster_recommend(prediction[0])

    show_recommendations(keyword)
    

    return {"top_products": cluster_terms}

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=fa869e6ded850e320a0128935d3adc38&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# def recommend(movie):
#     movie_index = movies[movies['title'] == movie].index[0]
#     distances = similarity[movie_index]
#     movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

#     recommend_movies = []
#     recommend_movies_posters = []
#     for i in movies_list:
#         movie_id = movies.iloc[i[0]].movie_id
#         recommend_movies.append(movies.iloc[i[0]].title)
#         # fetch poster from API
#         recommend_movies_posters.append(fetch_poster(movie_id))
#     return recommend_movies,recommend_movies_posters

# Define the route to the sentiment predictor
@app.post("/recommend_movie")
def predict_price(movie: str):
    names,posters = recommend(movie)

    return {
            "names": names,
            "posters": posters
           }