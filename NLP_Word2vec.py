import mailbox
import os
import random
import re
import string
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def getcharsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handleerror(errmsg, emailmsg,cs):
    print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",getcharsets(emailmsg))
    print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])

def getbodyfromemail(msg):
    body = None
    title = msg['subject']
    #Walk through the parts of the email to find the text body.
    if msg.is_multipart():
        for part in msg.walk():

            # If part is multipart, walk through the subparts.
            if part.is_multipart():

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True)
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif msg.get_content_type() == 'text/plain':
        body = msg.get_payload(decode=True)

   # No checking done to match the charset with the correct part.
    for charset in getcharsets(msg):
        try:
            body = body.decode(charset)
        except UnicodeDecodeError:
            handleerror("UnicodeDecodeError: encountered.",msg,charset)
        except AttributeError:
             handleerror("AttributeError: encountered" ,msg,charset)
    return [body,title]


directory = 'emails'
docs = []
emails = []
stop_words = set(stopwords.words('english') + ["paper", "papers", '2023', '2022', 'http', 'https', 'best', 'regards'])
counter = 0
titles = []

for thisemail in mailbox.mbox(r"C:\Users\Adham Hesham\Desktop\ML-news\allEmails.mbox"):
    body, title = getbodyfromemail(thisemail)
    if len(re.findall(r"(?=.*^((?!\[job\]).)*$)(?=.*^((?!\[jobs\]).)*$)(?=.*^((?!fellowship).)*$)(?=.*phd)",title.lower())) > 0 and body != None:
        titles.append(title)
        body = str(body)
        try:
            body = body[0: re.search("You received this message because you are subscribed to the Google Groups", body).start()]
        except:
            print(body)
        emails.append(body)
        word_tokens = word_tokenize(body)
        filtered_punc = []
        for word in word_tokens:
            if (word.lower() not in stop_words) and (word not in string.punctuation):
                filtered_punc.append(word)
        docs.append(filtered_punc)

'''for file in os.scandir(directory):
    if file.is_file():
        counter += 1
        with open(file.path, 'r') as fp:
            lines = ""
            start_reading = False
            for i, line in enumerate(fp):
                if 'Google Groups "Machine Learning News"' in line:
                    break
                if line.strip() == "":      # When the first empty line comes up start reading
                    start_reading = True
                if start_reading:
                    lines = lines + " " + line.strip()
        emails.append(lines)
        word_tokens = word_tokenize(lines)
        titles.append(os.path.splitext(file.name)[0])
        filtered_punc = []
        for word in word_tokens:
            if (word.lower() not in stop_words) and (word not in string.punctuation):
                filtered_punc.append(word)
        docs.append(filtered_punc)'''

model = Word2Vec(sentences=docs, vector_size=100, workers=1)


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

vectorized_docs = vectorize(docs, model=model)

def mbkmeans_clusters(X, k, mb, print_silhouette_values):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

'''Sum_of_squared_distances = []
K = range(2,100)
for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(vectorized_docs)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()'''

k = 10
clustering, cluster_labels = mbkmeans_clusters( X=vectorized_docs, k=k, mb=500, print_silhouette_values=True)

df_clusters = pd.DataFrame({
    "title": titles,
    "text": emails,
    "tokens": [" ".join(text) for text in docs],
    "cluster": cluster_labels
})
df_clusters.to_csv("Word2vec.csv")
print("Most representative terms per cluster (based on centroids):")
for i in range(k):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

'''test_cluster = 3
most_representative_docs = np.argsort(
    np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1)
)
for d in most_representative_docs[:3]:
    print(titles[d])
    print("-------------")'''

most_representative_docs = []
for i in range(k):
    most_representative_docs.append(np.argsort(
        np.linalg.norm(vectorized_docs - clustering.cluster_centers_[i], axis=1)
    )
    )
for d in most_representative_docs:
    print(titles[d[0]])
    print(titles[d[1]])
    print(titles[d[2]])
    print("-------------")
