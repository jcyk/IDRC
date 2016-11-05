import json
from collections import defaultdict

train_file = 'pdtb.train'

raw_train_data = [json.loads(line) for line in open(train_file).readlines()]

count_sense = defaultdict(int)
for d in raw_train_data:
	count_sense[d['Sense']]+=1
total = 0
for key,value in count_sense.iteritems():
	total += value
	print key,value
print 'total',len(count_sense),total

#print count_sense['Temporal.Asynchronous']
#print count_sense['Temporal.Synchrony']
'''
print count_sense['Contingency.Cause']
print count_sense['Contingency.Pragmatic cause']
print count_sense['Contingency.Condition']
print count_sense['Contingency.Pragmatic condition']
print count_sense['Comparison.Contrast']
print count_sense['Comparison.Pragmatic contrast']
print count_sense['Comparison.Concession']
print count_sense['Comparison.Pragmatic concession']
print count_sense['Expansion.Conjunction']
print count_sense['Expansion.Instantiation']
print count_sense['Expansion.Restatement']
print count_sense['Expansion.Alternative']
print count_sense['Expansion.Exception']
print count_sense['Expansion.List']



another_train_file = 'ano'
another_train_data = [json.loads(line) for line in open(another_train_file).readlines()]
idx = 0
for the,ano in zip(raw_train_data,another_train_data):
	idx+=1
	the_arg1 = the['Arg1'].replace(' ','')
	ano_arg1 = ano['Arg1']['RawText'].replace(' ','')
	if the_arg1 != ano_arg1:
		print idx,the['Where'],the['Sense']
		print 'Y: ',the['Arg1']
		print 'N: ',ano['Arg1']['RawText']
		break
'''

