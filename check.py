import json

# Read the first file and extract the ids
with open("infoAssessment22.json") as f1:
    data1 = json.load(f1)
    ids1 = set()
    for zone in data1["zones"]:
        for question in zone["questions"]:
            ids1.add(question["id"])

# Read the second file and extract the ids
with open("result.json") as f2:
    data2 = json.load(f2)
    ids2 = set()
    for zone in data2["zones"]:
        for question in zone["questions"]:
            ids2.add(question["id"])

# Check if the ids from the first file are contained in the second file
if ids1.issubset(ids2):
    print("All ids from file1 are contained in file2.")
else:
    print("Not all ids from file1 are contained in file2.")
