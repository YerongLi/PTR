import json
import random
from templating import get_temps
oldtemp = open('data/retacred/temp.txt', 'r')
newtemp = open('data/retacred/temp.1.txt', 'w')
path =  'data/retacred/test.txt'
print('start the loop')
candidate = {'per:employee_of', 'per:title', 'org:country_of_branch', 'per:city_of_death'}
candidate = {c : list() for c in candidate}
with open(path, "r") as f:
    features = []
    for line in f.readlines():
        line = line.rstrip()
        if len(line) > 0:
        	data = eval(line)
        	if data['relation'] in candidate:
        		candidate[data['relation']].append((data['h']['name'], data['t']['name']))
            # features.append(eval(line))  
print(candidate.keys())
for i in oldtemp.readlines():
    entries = i.strip().split("\t")
    if entries[1] in candidate:
    	if entries[1] == 'per:employee_of':
    		pairs = random.sample(candidate[entries[1]], 30)
    	else:
    		pairs = random.sample(candidate[entries[1]], 3)
    	for pair in pairs :
    		newtemp.write(i[:-1] + '\t'+ '\t'.join(pair) + '\n')
    else:
    	newtemp.write(i)

    	continue
