import re, nltk
from datasets import load_dataset
import json 
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
dataset = load_dataset("hugginglearners/netflix-shows", split="train")
df = pd.DataFrame(dataset)
#nltk.download('punkt')
def preprocess_text(text: str) -> str:
   
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    
    # remove stopwords
    tokens = nltk.word_tokenize(text)
    text = " ".join(tokens)
    text = text.lower().strip()
    
    return text

df['description'] = df['description'].apply(lambda text: preprocess_text(text))
df = df[df['description'] != '']

vectorizer = TfidfVectorizer(stop_words='english') 
# vectorizer the text documents 
vectorized_documents = vectorizer.fit_transform(df['description']) 
# reduce the dimensionality of the data using PCA 
pca = PCA(n_components=10) 
reduced_data = pca.fit_transform(vectorized_documents.toarray()) 
  
  
# cluster the documents using k-means 
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=5, 
                max_iter=500, random_state=42) 
kmeans.fit(vectorized_documents) 
  
  
# create a dataframe to store the results 
results = pd.DataFrame() 
results['document'] = df["description"]
results['cluster'] = kmeans.labels_ 
from matplotlib import colors as mcolors

tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
# plot the results 
colors = ['red', 'green','yellow','blue'] 
cluster = ['Not Sarcastic','Sarcastic'] 
for i in range(num_clusters): 
    plt.scatter(reduced_data[kmeans.labels_ == i, 0], 
                reduced_data[kmeans.labels_ == i, 1],  
                s=10, color=tableau_colors[i],  
                label=f' {i}') 
plt.legend() 
plt.show()
print(num_clusters)

