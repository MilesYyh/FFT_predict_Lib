import sys
from Bio import SeqIO

sequence = ""
#read and save sequences
for record in SeqIO.parse(sys.argv[1], "fasta"):
	for residue in record.seq:
		sequence+=residue

file_export = open("full_landscape.txt", 'w')

index_pos=1
for residue in sequence:

	for mutation in ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]:
		if residue != mutation:
			line = "A %s%d%s" % (residue, index_pos, mutation)
			file_export.write(line+"\n")
	index_pos+=1
file_export.close()


	