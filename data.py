import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
descriptions = data['description'].astype(str)  # Assuming 'description' is the column name

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return words

# Preprocess descriptions
processed_texts = [preprocess_text(desc) for desc in descriptions]

# Compute word frequencies
word_freq = {}
for text in processed_texts:
    for word in text:
        word_freq[word] = word_freq.get(word, 0) + 1

# Sort by frequency
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Visualizing the results
def plot_word_frequencies(word_freq, N):
    words, frequencies = zip(*word_freq[:N])  # Unpack word-frequency pairs into two lists
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {N} Words by Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Display top N words
N = 10
plot_word_frequencies(sorted_word_freq, N)
