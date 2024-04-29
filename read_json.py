import json

# Specify the path to JSON file
file_path = '/home/corina/Documents/computer_vision/final_proj/abstract_images/MultipleChoice_abstract_v002_train2015_questions.json'

# Open the JSON file for reading
with open(file_path, 'r') as file:
    # Load JSON data from the file into a Python dictionary
    data_train_abstract = json.load(file)

#print(data_train_abstract[0]['questions'])

# Key for which I want to find all possible values
target_key = 'task_type'

# Extracting just the questions
questions = [entry['question'] for entry in data_train_abstract['questions']]

print("end of file")