#nltk.download('stopwords')
#nltk.download('punkt')
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import os
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import mailbox
from sklearn.cluster import KMeans
#from transformers import
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

def sort_dict(dict):
    keys = list(dict.keys())
    values = list(dict.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_dict

directory = 'emails'
global_lines = []
stop_words = set(stopwords.words('english') + ["paper", "papers", '2023', '2022', 'http', 'https', 'best', 'regards', 'com', 'news', 'research'])
counter = 0
titles = []
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
        titles.append(os.path.splitext(file.name)[0])
        global_lines.append(lines)'''

for thisemail in mailbox.mbox(r"C:\Users\Adham Hesham\Desktop\ML-news\allEmails.mbox"):
    body, title = getbodyfromemail(thisemail)
    if len(re.findall(r"(?=.*^((?!\[job\]).)*$)(?=.*^((?!\[jobs\]).)*$)(?=.*^((?!fellowship).)*$)(?=.*phd)",title.lower())) > 0 :
        titles.append(title)
        body = str(body)
        try:
            body = body[0: re.search("You received this message because you are subscribed to the Google Groups",
                                     body).start()]
        except:
            print(body)
        global_lines.append(body)

'''vectorizer = TfidfVectorizer(max_features=5, stop_words=list(stop_words))
X = vectorizer.fit_transform(global_lines)
Sum_of_squared_distances = []
K = range(2,20)
for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(X)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()'''

vectorizer = TfidfVectorizer(max_features=10, stop_words=list(stop_words))
X = vectorizer.fit_transform(global_lines).todense()
df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
print(vectorizer.get_feature_names_out())
true_k = 7
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
labels = model.labels_
emails_cl = pd.DataFrame(list(zip(titles, labels)), columns=['title', 'cluster'])
print(emails_cl.sort_values(by=['cluster']))
print(X)
#emails_cl.to_csv("Tfidf.csv")
linkage_data = linkage(X, method='ward', metric='euclidean')
print(linkage_data)
dendrogram(linkage_data)

plt.show()
