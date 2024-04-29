import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
import pickle

# Ensure that NLTK's components are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# Function to find antonyms using WordNet
def find_antonym(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None

# Function to produce semantic negation of a sentence using NLTK
def semantic_negation(sentence, negated_sentences, unmodified_sentences, both):
    tokens = word_tokenize(sentence)  # Tokenize the sentence
    tagged = pos_tag(tokens)  # POS Tagging
    negated_sentence = []
    modified = False

    for word, tag in tagged:
        nltk_tag = tag[0].lower()  # 'J' for adjective, 'R' for adverb
        if nltk_tag in ['j', 'r']:
            antonym = find_antonym(word)
            if antonym:
                negated_sentence.append(antonym)
                modified = True
            else:
                negated_sentence.append(word)
        else:
            negated_sentence.append(word)

    result_sentence = ' '.join(negated_sentence)
    if modified:
        negated_sentences.append(result_sentence)
        both.append((sentence, result_sentence))
    else:
        unmodified_sentences.append(sentence)

# List of questions
#questions = [
   # "is the boy dangerously close to the fire?"
    # Add more questions
#]

if __name__ == "__main__":
    # Apply semantic negation

    # Loading the list from a file
    with open('questions_list.pkl', 'rb') as f:
        loaded_list = pickle.load(f)

    both = []
    negated_sentences = []
    unmodified_sentences = []
    for question in loaded_list:
        semantic_negation(question, negated_sentences, unmodified_sentences, both)

    final = set(both)

print("end")