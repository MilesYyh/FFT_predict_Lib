import pandas as pd
import sys

dataset = pd.read_csv(sys.argv[1])

values = [value for value in dataset['Predict']]

#make count by interval
cont_low_0 =0
cont_0_1 = 0
cont_1_3 = 0
cont_hig_3 =0

for value in values:
	if value<=0:
		cont_low_0+=1
	elif value >0 and value<=1:
		cont_0_1+=1
	elif value>1 and value<=3:
		cont_1_3+=1
	else:
		cont_hig_3+=1

print(cont_low_0)
print(cont_0_1)
print(cont_1_3)
print(cont_hig_3)
