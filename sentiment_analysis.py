#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk


# In[ ]:


import pandas as pd

df = pd.read_csv('/content/Reviews.csv', on_bad_lines='skip')
print(df.head())


# In[ ]:


ax = df['Score'].value_counts().sort_index()     .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# In[ ]:


example = df['Text'][50]
print(example)
df = df.head(500)
print(df.shape)


# In[ ]:


import nltk
nltk.download('punkt_tab')
tokens = nltk.word_tokenize(example)
tokens[:10]


# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# tagged = nltk.pos_tag(tokens)
# tagged[:10]

# In[ ]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[ ]:



nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# In[ ]:



sia.polarity_scores('This is the worst thing ever happened.')


# In[ ]:


sia.polarity_scores('I am so excited today!')


# In[ ]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    if isinstance(text, str):
      res[myid] = sia.polarity_scores(text)
    else:
      pass


# In[ ]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[ ]:


vaders.head()


# In[ ]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[ ]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[ ]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[ ]:


print(example)
sia.polarity_scores(example)


# In[ ]:


encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[ ]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[ ]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[ ]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
results_df.head()


# In[ ]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# In[ ]:


results_df.query('Score == 1')     .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[ ]:


results_df.query('Score == 1')     .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[ ]:


results_df.query('Score == 5')     .sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[ ]:


results_df.query('Score == 5')     .sort_values('vader_neg', ascending=False)['Text'].values[0]


# In[ ]:


from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[ ]:


sent_pipeline('I love sentiment analysis!')


# In[ ]:


sent_pipeline('this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault')

