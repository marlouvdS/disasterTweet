import pandas as pd
from nltk import PorterStemmer, re
from nltk.corpus import stopwords

def preprocessing (df):
    corpus = []
    pstem = PorterStemmer()
    for i in range(df['text'].shape[0]):
        # Remove unwanted words
        text = re.sub("[^a-zA-Z]", ' ', df['text'][i])
        # Transform words to lowercase
        text = text.lower()
        text = text.split()
        # Remove stopwords then Stemming it
        text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        # Append cleaned tweet to corpus
        corpus.append(text)
    print("Corpus created successfully")

    return corpus

def createDict(corpus):
    # Create dictionary
    uniqueWords = {}
    for text in corpus:
        for word in text.split():
            if (word in uniqueWords.keys()):
                uniqueWords[word] += 1
            else:
                uniqueWords[word] = 1

    # Convert dictionary to dataFrame
    uniqueWords = pd.DataFrame.from_dict(uniqueWords, orient='index', columns=['WordFrequency'])
    uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
    print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))
    print(uniqueWords.head(10))
    return uniqueWords