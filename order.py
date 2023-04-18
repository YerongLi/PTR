import json

# Load the first file
with open('infoAssessment22.json', 'r') as f1:
    data1 = json.load(f1)

# Load the second file
with open('result.json', 'r') as f2:
    data2 = json.load(f2)

# Create a dictionary of the `zones:title` in the first file
title_order = {}
for i, zone in enumerate(data1['zones']):
    title_order[zone['title']] = i

# Sort the `zones` in the second file based on the order in the first file
data2['zones'] = sorted(data2['zones'], key=lambda x: title_order[x['title']])

# Write the reordered `zones` to the second file
with open('result.json', 'w') as f2:
    json.dump(data2, f2, indent=4)
