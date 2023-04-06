import json
from templating import get_temps
oldtemp = open('data/retacred/temp.txt', 'r')
newtemp = open('data/retacred/temp.1.txt', 'w')
if features is None:
    self.args = get_args()
    with open(path+"/" + name, "r") as f:
        features = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
            	print(dir(eval(line)))
                # features.append(eval(line))  
for i in oldtemp.readlines():
    entries = i.strip().split("\t")
    if entries[2] == 'per:':
    	continue
