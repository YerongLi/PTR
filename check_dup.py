import json

# Load the input file
with open('result.json', 'r') as f:
    data = json.load(f)

# Create a set to keep track of the question IDs
question_ids = set()

# Loop through the zones and questions
for zone in data['zones']:
    for question in zone['questions']:
        # Check if the question ID is already in the set
        if question['id'] in question_ids:
            print(f'Duplicated question ID: {question["id"]}')
        else:
            # Add the question ID to the set
            question_ids.add(question['id'])
