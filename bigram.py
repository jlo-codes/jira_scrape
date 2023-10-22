import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import bigrams, trigrams
import string
import matplotlib.pyplot as plt

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

# Extract bigrams and trigrams
bigram_freq = {}
trigram_freq = {}
for text in processed_texts:
    for bigram in bigrams(text):
        bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    for tri in trigrams(text):
        trigram_freq[tri] = trigram_freq.get(tri, 0) + 1

# Sort by frequency
sorted_bigram_freq = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
sorted_trigram_freq = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)

# Visualizing the results
def plot_ngram_frequencies(ngram_freq, N, ngram):
    words, frequencies = zip(*ngram_freq[:N])
    words = [' '.join(word) for word in words]
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies, color='blue')
    plt.xlabel(f'{ngram}s')
    plt.ylabel('Frequency')
    plt.title(f'Top {N} {ngram.capitalize()}s by Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

N = 10
plot_ngram_frequencies(sorted_bigram_freq, N, "bigram")
plot_ngram_frequencies(sorted_trigram_freq, N, "trigram")
