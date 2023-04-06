import json
from templating import get_temps
oldtemp = open('data/retacred/temp.txt', 'r')
newtemp = open('data/retacred/temp.1.txt', 'w')
testfile = json.load(open('data/retacred/test.txt', 'w')
for i in oldtemp.readlines():
    entries = i.strip().split("\t")
    if entries[2] == 'per:':
    	continue
