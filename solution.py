"""
Lab 04 - Bag of Words & Cosine Similarity
Task 1: Bag of Words Matrix
Task 2: Document Search Engine using Cosine Similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# TASK 1 - Bag of Words Matrix
# ============================================================
print("=" * 60)
print("TASK 1 - Bag of Words Matrix")
print("=" * 60)

corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

vectorizer1 = CountVectorizer(stop_words='english')
bow_matrix = vectorizer1.fit_transform(corpus)

feature_names = vectorizer1.get_feature_names_out()

df_bow = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
print(df_bow)

# ============================================================
# TASK 2 - Document Search Engine
# ============================================================
print("\n" + "=" * 60)
print("TASK 2 - Document Search Engine (Cosine Similarity)")
print("=" * 60)

documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

query = ["machine learning algorithms for data"]

vectorizer2 = CountVectorizer(stop_words='english')
doc_vectors = vectorizer2.fit_transform(documents)
query_vector = vectorizer2.transform(query)

similarities = cosine_similarity(query_vector, doc_vectors)

results = pd.DataFrame({
    "Document": documents,
    "Similarity Score": similarities.flatten()
})

results = results.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)

print(results.to_string(index=False))
