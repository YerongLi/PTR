import json

# read the first file
with open('infoAssessment.json', 'r') as f1:
    data1 = json.load(f1)

# read the second file
with open('infoAssessment22.json', 'r') as f2:
    data2 = json.load(f2)

# merge the data
merged_data = data1.copy()
merged_data['zones'] += data2['zones']

# remove duplicates
unique_questions = []
for zone in merged_data['zones']:
    unique_questions_zone = []
    for question in zone['questions']:
        if question['id'] not in [q['id'] for q in unique_questions_zone]:
            unique_questions_zone.append(question)
    zone['questions'] = unique_questions_zone
    unique_questions += unique_questions_zone

# group similar questions
grouped_questions = []
for question in unique_questions:
    if question['id'] not in [q['id'] for q in grouped_questions]:
        grouped_questions.append(question)
    else:
        idx = [q['id'] for q in grouped_questions].index(question['id'])
        grouped_questions[idx]['points'] += question['points']

# update the merged data with the grouped questions
for zone in merged_data['zones']:
    for i, question in enumerate(zone['questions']):
        idx = [q['id'] for q in grouped_questions].index(question['id'])
        zone['questions'][i] = grouped_questions[idx]

# write the merged data to a file
with open('merged_file.json', 'w') as f:
    json.dump(merged_data, f, indent=4)
