import sys
import glob
import os
import pandas as pd

path_input = "result_landscape/result_landscape/"

matrix_data = []

for i in range(20, 3041, 20):
	
	#untar file
	path_search = path_input+"small_file_"+str(i)+"/"	
	name_file = glob.glob(path_search+"*.tar")	
	command = "tar -xvf %s -C %s" % (name_file[0], path_search)
	print(command)
	os.system(command)

	#get name file for make lecture
	name_dataset = name_file[0].split("/")[-1].split(".")
	name_dataset = name_dataset[0]+"."+name_dataset[1].split("_")[0]+"_sdm_output.csv"
	
	dataset = pd.read_csv(path_search+name_dataset, header=None)
	
	for i in range(len(dataset)):
		matrix_data.append(dataset.iloc[i])

	command = "rm %s*.csv" % path_search
	os.system(command)	

matrix_data = pd.DataFrame(matrix_data)
name_output = sys.argv[1]
matrix_data.to_csv(name_output, index=False, sep=",")