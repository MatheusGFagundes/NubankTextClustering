#!/usr/bin/env python
# coding: utf-8

# 
# In this kernel, we will get complaints from 'Reclame Aqui' to identify pattern and frequency of words and clustering through unsupervised learning algorthim.
# 
# the corpus has 10010 complaints from 'Reclame Aqui' that regards roughly a interval of one year (04/2018 - 04/2019)

# importing complaints txt files from reclame aqui

# In[71]:


import os
files = []
for i in os.listdir("corpus"):
    if i.endswith('.txt'):
        files.append(open("corpus/" + i).read())
print(len(files))


# Pre-processing
# 
# In this step, in order to get information from texts, We need to pre-process these texts which means removing and cleaning all of what it is not main to detect similarities,  such as propositions, articles, very common words. in other words, keeping only the essential.
# Afterward, we will transform the texts into a CountVectorize(count term per text) and TfidfVectorizer(term frequency–inverse document frequency), both to represent the frequency of words in each text  

# All number and special characters, except accents in the Portuguese language, will be removed and set in lower-case.
# 
# afterward, we will split the texts into words, and check each one if it is stopword, and remove it

# In[72]:


from nltk.corpus import stopwords
# the first fiftyth caracters
before = files[0][0:50]

print(before)
import re
for index,complaint in enumerate(files):
    complaint = re.sub('[^A-Za-z,"áéíóúâêîôûãõçà"]',' ',complaint)
    complaint = complaint.replace(',', ' ')
    complaint = complaint.lower()
    complaint = complaint.split()
    complaint = [word.strip() for word in complaint if not word.strip() in stopwords.words('portuguese') and len(word.strip()) > 2 and word.strip() != "nubank" ]
    files[index] = complaint
after = files[0][0:50]
print(after)


# we will implement Text Normalization by stemming that means putting all words in their own stem(root) and save in dictionary the previous variant before applying Normalization.

# In[73]:


import nltk 
from nltk.stem import RSLPStemmer
stemmer = RSLPStemmer()
all_words = {}
before = files[0]
print(before)
print("")
for x in range(len(files)):
   for y,word in enumerate(files[x]):
       new = stemmer.stem(word)
       if(new in all_words):
           if(word in all_words[new]):
               all_words[new][word]+=1
           else:
               all_words[new][word] = 1
       else:
           all_words[new] = {}
           all_words[new][word] = 1
       files[x][y] = new
after = files[0]
print(after)


# we will restore most frequent versions of each word in order to label them to better visualization in wordclouds

# In[74]:


import operator
for x in range(len(files)):
   for y,word in enumerate(files[x]):
       files[x][y] = sorted(all_words[word].items(), key=operator.itemgetter(1), reverse = True)[0][0]

corpus = files.copy()

for x in range(len(corpus)):
    corpus[x] = ' '.join(corpus[x]) 
    


# the texts will be converted in a CountVectorizer and TfidfVectorizer(term frequency–inverse document frequency), both to represent the frequency of words in each text, and words with less than 0.01 will be removed, such as very rare words, misspelled, names, etc.

# In[75]:


import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.01)
matrizcount = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(min_df=0.01)
matrizfrequency = tf.fit_transform(corpus).toarray() 




# To obtain a notion of which words represent each cluster, we have to calculate words frequency mean per cluster.
# 
# 

# General word cloud from CountVectorizer.

# In[76]:


import numpy as np
from os import path
from PIL import Image
nubank_coloring = np.array(Image.open("nu-icon.png"))
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud,ImageColorGenerator
get_ipython().run_line_magic('matplotlib', 'notebook')

words = pd.DataFrame(matrizcount)
words.columns =  cv.get_feature_names()
words = words.mean()
words.head()

wordcloud = wordcloud = WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words)

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()








# General word cloud from TfidfVectorizer.

# In[77]:


import numpy as np
from os import path
from PIL import Image
nubank_coloring = np.array(Image.open("nu-icon.png"))
import matplotlib.pyplot as pl
import pandas as pd
from wordcloud import WordCloud,ImageColorGenerator
words = pd.DataFrame(matrizfrequency)
words.columns =  tf.get_feature_names()
words = words.mean()
words.head()

wordcloud = wordcloud = WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words)

image_colors = ImageColorGenerator(nubank_coloring)
pl.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
pl.axis("off")
pl.show()


# Processing text
# 
# we will use k-means, which is an unsupervised algorithm, to find  clusters of texts that have words in common. 
# 
# in order to find out the best K and which  frequency representation matrix is better, we will run Kmeans with several K 
# 
# Elbow and Average silhouette method will be used to measure the quality of cluster and find the best k. 

# Running kmeans 2 to 20 with CountVectorizer

# In[103]:


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
wcss= []
silh = []
for k in range(2,20):
    model = KMeans(n_clusters=k, init='k-means++',max_iter=300,n_init= 50)
    model.fit(matrizcount)
    wcss.append(model.inertia_)    
    silhouette_avg = silhouette_score(matrizcount, model.labels_)
    silh.append(silhouette_avg)


# In[104]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(range(2,20),wcss)
axes[0].set_title("Elbow method")
axes[1].plot(range(2,20),silh)
axes[1].set_title("Average silhouette method")
plt.show()


# Running kmeans 2 to 20 with TfidfVectorizer

# In[83]:


wcss= []
silh = []
for k in range(2,20):
    model = KMeans(n_clusters=k, init='k-means++',max_iter=300,n_init= 50)
    model.fit(matrizfrequency)
    wcss.append(model.inertia_)    
    silhouette_avg = silhouette_score(matrizfrequency, model.labels_)
    silh.append(silhouette_avg)


# In[85]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(range(2,20),wcss)
axes[0].set_title("Elbow method")
axes[1].plot(range(2,20),silh)
axes[1].set_title("Average silhouette method")
plt.show()


# as the results with TfidfVectorizer it was considerably better, it will be chosen to use in our cluster model (k-means) with 
# K=8

# In[87]:


k_chosen = 8;
model = KMeans(n_clusters=k_chosen, init='k-means++',max_iter=1000,n_init= 100)
model.fit(matrizfrequency)

every text is classified within 1(index 0) to 8 (index 7)
# In[90]:


words = pd.DataFrame(matrizfrequency)
words.columns = tf.get_feature_names()
words['cluster'] = model.labels_
words.head()


# in order to visualize the content of the texts for each cluster, we will aggroup and take a mean of the  words frequency in each cluster

# In[91]:


words = words.groupby('cluster').mean()
words.head(8)


# Word Clouds of each cluster

# In[92]:



wordcloud = WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[0,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 1')
plt.show()


# In[95]:



wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[1,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 2')
plt.show()


# In[96]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[2,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 3')
plt.show()


# In[97]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[3,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 4')
plt.show()


# In[98]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[4,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 5')
plt.show()


# In[99]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[5,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 6')
plt.show()


# In[100]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[6,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 7')
plt.show()


# In[102]:


wordcloud =  WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(words.iloc[7,:])

image_colors = ImageColorGenerator(nubank_coloring)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title('Cluster 8')
plt.show()

