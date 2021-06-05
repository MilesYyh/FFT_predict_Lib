import sys
import os 
import glob

list_files = glob.glob("result_landscape/*.txt")

for file in list_files:
	print("Process file: ", file)

	command = "python3 sdm_service.py %s" % file
	os.system(command)
	