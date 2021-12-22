# Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import string
import re
from wordcloud import WordCloud, STOPWORDS
import streamlit as st


# Loading the dataset
reviews_dataset = pd.read_csv("C:/Users/Dell/Downloads/mussorie_reviews.csv", header=None, names=['Review', 'Date of stay', 'Rating'])
reviews_dataset.head()

# Checking information about the dataset
reviews_dataset.info()

rows = reviews_dataset.shape[0]

print('There are a total of {} entries in the dataset.'.format(rows))

# Dropping all the rows with null values
reviews_dataset.dropna(inplace=True)

# Dropping duplicate rows
reviews_dataset.drop_duplicates(inplace=True)

# Checking information for the dataset after removing null value rows and duplicate rows
reviews_dataset.info()

# Extracting rating value from the unclean rating column
reviews_dataset["Rating"] = reviews_dataset["Rating"].apply(lambda x: re.findall("\d", x)[0])

# Checking percentage of rows for each rating
reviews_dataset["Rating"].value_counts(normalize = True)

# Mapping positive, neutral and negative to rating values 
reviews_dataset["Sentiment"] = reviews_dataset["Rating"].map({"5":"Positive", "4":"Positive", "3":"Neutral", "2":"Negative", "1":"Negative"})


# Creating a new column year of stay from our existing column date of stay
reviews_dataset["Year of stay"] = reviews_dataset["Date of stay"].apply(lambda x: "".join(re.findall("\d", x)))

# Creating a month of stay column from date of stay column
reviews_dataset["Month of stay"] = reviews_dataset["Date of stay"].apply(lambda x: x.split(" ")[3])

# Dropping all rows for which year is less than 2018
reviews_dataset.drop(reviews_dataset[reviews_dataset["Year of stay"] < '2018'].index, axis=0, inplace=True)

# Dropping all neutral sentiment rows
reviews_dataset.drop(reviews_dataset[reviews_dataset["Sentiment"] == "Neutral"].index, axis=0, inplace=True)

# Shuffling the dataset
reviews_dataset = reviews_dataset.sample(frac = 1)

# Dropping date of stay and rating column from our dataset
reviews_dataset.drop(['Date of stay', 'Rating'], axis=1, inplace=True)

# Creating a function to preprocess data
def preprocess_data(text):
    """
    Returns text after removing numbers, punctuations, urls, emojis, html tags and lowercasing all the words in given text.
    """
    text = re.sub(r'[0-9]+', '', str(text))   # removing numbers
    text = re.sub(r'[^\w\s]', '', str(text))   # removing punctuations
    text = " ".join(x.lower() for x in text.split())  # lower casing the text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # removing urls
    text = re.sub(r'<.*?>', '', text) # removing html tags
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = " ".join(x for x in text.split() if x not in STOPWORDS)
    return text

# Applying preprocess_data function to review column
reviews_dataset["Preprocessed_reviews"] = reviews_dataset["Review"].apply(lambda x: preprocess_data(x))

# Word count
reviews_dataset["Word_count"] = reviews_dataset["Review"].apply(lambda x: len(str(x).split()))

# Unique word count
reviews_dataset["Unique_word_count"] = reviews_dataset["Review"].apply(lambda x: len(set(str(x).split())))

# Number of stop words in review
reviews_dataset["Stopword_count"] = reviews_dataset["Review"].apply(lambda x: len([w for w in str(x).lower().split() if w not in STOPWORDS]))

# mean word length
reviews_dataset["Mean_word_length"] = reviews_dataset["Review"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#char count
reviews_dataset["Char_count"] = reviews_dataset["Review"].apply(lambda x: len(str(x)))

# punctuation count
reviews_dataset["Punctuation_count"] = reviews_dataset["Review"].apply(lambda x: len([p for p in str(x) if p in string.punctuation]))


# Creating a pie chart showing target distribution in dataset
labels = 'Positive', 'Negative'
sizes = reviews_dataset['Sentiment'].value_counts()
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)


# Funtion for generating ngrams
def generate_ngrams(text, n_gram=1):
    """
    Returns ngrams for the given text.
    
    Parameters:
    text : for which we want ngrams
    n_gram : value for ngram
    """
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


# Creating and plotting unigrams
# Creating a default dictionary for positive and negative unigrams
positive_unigrams = defaultdict(int)
negative_unigrams = defaultdict(int)

# Number of ngrams we want
N = 25

# Loop for updating value of positive_unigrams
for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Positive"]:
    for word in generate_ngrams(review):
        positive_unigrams[word] += 1

for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Negative"]:
    for word in generate_ngrams(review):
        negative_unigrams[word] += 1
        
# Creating dataframes using default dictionaries
df_positive_unigrams = pd.DataFrame(sorted(positive_unigrams.items(), key=lambda x: x[1])[::-1])
df_negative_unigrams = pd.DataFrame(sorted(negative_unigrams.items(), key=lambda x: x[1])[::-1])

        
fig2, axes2 = plt.subplots(ncols=2, figsize=(24, 10))
plt.tight_layout(pad=4.0)

# Plotting positive and negative unigrams dataset
sns.despine()
sns.barplot(y = df_positive_unigrams[0].values[:N], x = df_positive_unigrams[1].values[:N], ax=axes2[0], color='#72EEA4')
sns.barplot(y = df_negative_unigrams[0].values[:N], x = df_negative_unigrams[1].values[:N], ax=axes2[1], color='#F37D67')

axes2[0].set_title(f'Top {N} most common unigrams in Positive Reviews', fontsize=15)
axes2[1].set_title(f'Top {N} most common unigrams in Negative Reviews', fontsize=15)

st.pyplot(fig2)


# Creating and plotting bigrams
# Creating a default dictionary for positive and negative bigrams
positive_bigrams = defaultdict(int)
negative_bigrams = defaultdict(int)

# Number of ngrams we want
N = 25

# Loop for updating values of bigrams in default dictionary
for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Positive"]:
    for word in generate_ngrams(review, n_gram=2):
        positive_bigrams[word] += 1

for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Negative"]:
    for word in generate_ngrams(review, n_gram=2):
        negative_bigrams[word] += 1
        
# Creating a dataset using default dictionaries
df_positive_bigrams = pd.DataFrame(sorted(positive_bigrams.items(), key=lambda x: x[1])[::-1])
df_negative_bigrams = pd.DataFrame(sorted(negative_bigrams.items(), key=lambda x: x[1])[::-1])
        
fig3, axes3 = plt.subplots(ncols=2, figsize=(24, 10))
plt.tight_layout(pad=4.0)
 
# Plotting both positive and negative bigrams
sns.despine()
sns.barplot(y = df_positive_bigrams[0].values[:N], x = df_positive_bigrams[1].values[:N], ax=axes3[0], color='#72EEA4')
sns.barplot(y = df_negative_bigrams[0].values[:N], x = df_negative_bigrams[1].values[:N], ax=axes3[1], color='#F37D67')

axes3[0].set_title(f'Top {N} most common bigrams in Positive Reviews', fontsize=15)
axes3[1].set_title(f'Top {N} most common bigrams in Negative Reviews', fontsize=15)

st.pyplot(fig3)


# Creating and plotting trigrams
# Creating default dictionary for positive and negative trigrams
positive_trigrams = defaultdict(int)
negative_trigrams = defaultdict(int)

# Number of ngrams we want
N = 25

# Loop for updating default dictionaries 
for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Positive"]:
    for word in generate_ngrams(review, n_gram=3):
        positive_trigrams[word] += 1

for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Negative"]:
    for word in generate_ngrams(review, n_gram=3):
        negative_trigrams[word] += 1
        
# Creating a dataframe using dictionaries
df_positive_trigrams = pd.DataFrame(sorted(positive_trigrams.items(), key=lambda x: x[1])[::-1])
df_negative_trigrams = pd.DataFrame(sorted(negative_trigrams.items(), key=lambda x: x[1])[::-1])
        
fig4, axes4 = plt.subplots(ncols=2, figsize=(24, 10))
plt.tight_layout(pad=4.0)
     
# Plotting positive and negative trigrams
sns.despine()
sns.barplot(y = df_positive_trigrams[0].values[:N], x = df_positive_trigrams[1].values[:N], ax=axes4[0], color='#72EEA4')
sns.barplot(y = df_negative_trigrams[0].values[:N], x = df_negative_trigrams[1].values[:N], ax=axes4[1], color='#F37D67')

axes4[0].set_title(f'Top {N} most common trigrams in Positive Reviews', fontsize=15)
axes4[1].set_title(f'Top {N} most common trigrams in Negative Reviews', fontsize=15)

st.pyplot(fig4)


# Plotting number of reviews in each month
fig5 = plt.figure(figsize = (24, 5))
month =['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sns.despine()
sns.countplot(x="Month of stay", data=reviews_dataset, order=month, color='#62DCEC');
st.pyplot(fig5)
"""**Month of January and October has the highest number of reviews so we can say that number of people staying in hotel in these month is more compared to other months. Month of April has the  lowest number of reviews.**"""


# Plotting number of reviews for each year
fig6 = plt.figure(figsize = (24, 5))
years = ['2018', '2019', '2020', '2021']
sns.despine()
sns.countplot(x="Year of stay", data=reviews_dataset, order=years, color='#62DCEC');
st.pyplot(fig6)
"""**2019 has the highest number of reviews whereas 2020 has the lowest because of global pandemic.**"""

# Plotting word count for reviews
fig6, axes6 = plt.subplots(ncols = 2, figsize=(24, 6))

sns.despine()
sns.histplot(ax=axes6[0], x="Word_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Positive'], color='#72EEA4');
sns.histplot(ax=axes6[1], x="Word_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Negative'], color='#F37D67');

axes6[0].set_title("Positive reviews", fontsize=16)
axes6[1].set_title("Negative reviews", fontsize=16)

axes6[0].set_ylabel("Number of reviews", fontsize=12)
axes6[1].set_ylabel("Number of reviews", fontsize=12)
axes6[0].set_xlabel("Word count in review", fontsize=12)
axes6[1].set_xlabel("Word count in review", fontsize=12)

fig6.suptitle("Word count for reviews", fontsize=20)
st.pyplot(fig6)

# Plotting unique word count for reviews
fig7, axes7 = plt.subplots(ncols = 2, figsize=(24, 6))

sns.despine()
sns.histplot(ax=axes7[0], x="Unique_word_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Positive'], color='#72EEA4');
sns.histplot(ax=axes7[1], x="Unique_word_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Negative'], color='#F37D67');

axes7[0].set_title("Positive reviews", fontsize=16)
axes7[1].set_title("Negative reviews", fontsize=16)

axes7[0].set_ylabel("Number of reviews", fontsize=12)
axes7[1].set_ylabel("Number of reviews", fontsize=12)
axes7[0].set_xlabel("Unique word count in review", fontsize=12)
axes7[1].set_xlabel("Unique word count in review", fontsize=12)

fig7.suptitle("Unique word count for reviews", fontsize=20)
st.pyplot(fig7)


# Plotting stopword count for reviews
fig8, axes8 = plt.subplots(ncols = 2, figsize=(24, 6))

sns.despine()
sns.histplot(ax=axes8[0], x="Stopword_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Positive'], color='#72EEA4');
sns.histplot(ax=axes8[1], x="Stopword_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Negative'], color='#F37D67');

axes8[0].set_title("Positive reviews", fontsize=16)
axes8[1].set_title("Negative reviews", fontsize=16)

axes8[0].set_ylabel("Number of reviews", fontsize=12)
axes8[1].set_ylabel("Number of reviews", fontsize=12)
axes8[0].set_xlabel("Stopword count in review", fontsize=12)
axes8[1].set_xlabel("Stopword count in review", fontsize=12)

fig8.suptitle("Stopword count for reviews", fontsize=20)
st.pyplot(fig8)

# Plotting character count for reviews
fig9, axes9 = plt.subplots(ncols = 2, figsize=(24, 6))

sns.despine()
sns.histplot(ax=axes9[0], x="Char_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Positive'], color='#72EEA4');
sns.histplot(ax=axes9[1], x="Char_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Negative'], color='#F37D67');

axes9[0].set_title("Positive reviews", fontsize=16)
axes9[1].set_title("Negative reviews", fontsize=16)

axes9[0].set_ylabel("Number of reviews", fontsize=12)
axes9[1].set_ylabel("Number of reviews", fontsize=12)
axes9[0].set_xlabel("Character count in review", fontsize=12)
axes9[1].set_xlabel("Character count in review", fontsize=12)

fig9.suptitle("Character count for reviews", fontsize=20)
st.pyplot(fig9)


# Plotting punctuation count for reviews
fig22, axes22 = plt.subplots(ncols = 2, figsize=(24, 6))

sns.despine()
sns.histplot(ax=axes22[0], x="Punctuation_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Positive'], color='#72EEA4');
sns.histplot(ax=axes22[1], x="Punctuation_count", data=reviews_dataset[reviews_dataset['Sentiment']=='Negative'], color='#F37D67');

axes22[0].set_title("Positive reviews", fontsize=16)
axes22[1].set_title("Negative reviews", fontsize=16)
axes22[0].set_xscale('log')

axes22[0].set_ylabel("Number of reviews", fontsize=12)
axes22[1].set_ylabel("Number of reviews", fontsize=12)
axes22[0].set_xlabel("Punctuation count in review", fontsize=12)
axes22[1].set_xlabel("Punctuation count in review", fontsize=12)

fig22.suptitle("Punctuation count for reviews", fontsize=20)
st.pyplot(fig22)


# Plotting a wordcloud for positive reviews
positive = " ".join(review for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Positive"])
wordcloud = WordCloud(background_color='white', stopwords = STOPWORDS, max_words=500, max_font_size=40, random_state=42, colormap='Greens').generate(positive)
fig33 = plt.figure(figsize=(15, 10))
plt.title("Most common words in positive reviews")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(fig33)


# Plotting wordcloud for negative reviews
negative = " ".join(review for review in reviews_dataset['Preprocessed_reviews'][reviews_dataset['Sentiment']=="Negative"])
wordcloud = WordCloud(background_color='white', stopwords = STOPWORDS, max_words=500, max_font_size=40, random_state=42, colormap = "Reds").generate(negative)
fig44 = plt.figure(figsize=(15, 10))
plt.title("Most common words in negative reviews")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(fig44)
