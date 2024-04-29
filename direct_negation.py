import json
import pickle
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
def directly_negate_question(question):
    words = question.split()
    first_verb = words[0].lower()
    negations = {'is': "isn't", 'are': "aren't", 'can': "can't", 'do': "don't", 'does': "doesn't", 'did': "didn't"}

    if first_verb in negations:
        negated_verb = negations[first_verb]
        words[0] = negated_verb
        negated_question = ' '.join(words)
    else:
        words.insert(1, 'not')
        negated_question = ' '.join(words)

    return negated_question

nltk.download('averaged_perceptron_tagger')

def directly_negate_question_with_handling(question):

    words = word_tokenize(question)  # Tokenizing the question into words
    tagged = pos_tag(words)  # Tagging each word with its part of speech

    negations = {'is': "isn't", 'are': "aren't", 'can': "can't", 'do': "don't", 'does': "doesn't", 'did': "didn't"}
    verb_index = next((i for i, (word, tag) in enumerate(tagged) if tag.startswith('VB')), None)

    if verb_index is not None:
        verb = words[verb_index].lower()
        if verb in negations:
            words[verb_index] = negations[verb]
        else:
            words.insert(verb_index + 1, 'not')
    else:
        # Fallback if no verb is found
        words.insert(1, 'not')

    negated_question = ' '.join(words)
    return negated_question

def get_question_by_image_id(data, image_id):

    for item in data['questions']:
        if item['image_id'] == image_id:
            return item
    return None

def save_dict_to_file(dictionary, file_path):
    with open(file_path, 'wb') as file:
        # Use pickle to serialize the dictionary and save it to the file
        pickle.dump(dictionary, file)

if __name__ == "__main__":

    # path to JSON file
    file_path = '/home/corina/Documents/computer_vision/final_proj/abstract_images_balanced_binary/OpenEnded_abstract_v002_train2017_questions.json'

    # Open the JSON file for reading
    with open(file_path, 'r') as file:
        # Load JSON data from the file into a Python dictionary
        data_train_abstract = json.load(file)

    questions = data_train_abstract['questions']

    # Persist questions list for train
    save_dict_to_file(questions, 'questions_baseline.pkl')

    questions_list = []
    for question in questions:
        questions_list.append(question['question'])

    # persist list of questions
    with open('questions_list.pkl', 'wb') as f:
        pickle.dump(questions_list, f)

    item = get_question_by_image_id(data_train_abstract, 87)

    print(item)

    # Loop through each question dictionary in the list
    for question_dict in data_train_abstract['questions']:
        # Apply the negation function to the 'question' text
        negated_question = directly_negate_question(question_dict['question'])
        # Add the negated question as a new key-value pair in the dictionary
        question_dict['question_negated'] = negated_question

    # Now, 'data' contains the original questions along with their negated versions
    print(data_train_abstract['questions'][:5])



