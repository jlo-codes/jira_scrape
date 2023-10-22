import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim import corpora, models
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Initialize lemmatizer and stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load data from Excel
file_path = r"C:\Users\tumin\OneDrive\Desktop\Code\jira_fake.xlsx"
data = pd.read_excel(file_path)
descriptions = data['description'].astype(str)

def preprocess_text(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

processed_texts = [preprocess_text(desc) for desc in descriptions]

# Prepare data for LDA
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Train LDA model
num_topics = 5  # Adjust as needed
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Display topics
topics = lda_model.print_topics(num_words=5)  # Adjust the number of words as needed
for topic in topics:
    print(topic)

# For visualization, you might also want to consider the `pyLDAvis` library which provides a great way to visualize topic models.
# Visualizing topics
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)