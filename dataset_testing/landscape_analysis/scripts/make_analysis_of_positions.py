import pandas as pd
import sys

dataset = pd.read_csv(sys.argv[1])

#pos = dataset['MUTATION'][1000][1:-1]
'''
mutation_list = dataset['MUTATION']
positions = [mutation[1:-1] for mutation in mutation_list]
positions = list(set(positions))

matrix_export = []

for position in positions:

	print("Check position: ", position)
	cont_reduce = 0
	cont_increase = 0
	cont_neutra = 0

	for i in range(len(dataset)):

		pos = dataset['MUTATION'][i][1:-1]

		if pos == position:

			if dataset['Predict'][i]<=-1:

				cont_reduce+=1

			elif dataset['Predict'][i]>=1:
				
				cont_increase+=1

			else:
				cont_neutra+=1

	row = [position, cont_reduce, cont_neutra, cont_increase]
	matrix_export.append(row)

data_export = pd.DataFrame(matrix_export, columns=["position", "cont_reduce", "cont_neutra", "cont_increase"])
data_export.to_csv(sys.argv[2], index=False, sep=",")
'''

max_mutation = max(dataset['Predict'])
min_mutation = min(dataset['Predict'])

#search the mutations for min and max values

max_mutations = [dataset['MUTATION'][i] for i in range(len(dataset)) if dataset['Predict'][i] == max_mutation]
min_mutations = [dataset['MUTATION'][i] for i in range(len(dataset)) if dataset['Predict'][i] == min_mutation]

print(max_mutation)
print(max_mutations)

print(min_mutation)
print(min_mutations)