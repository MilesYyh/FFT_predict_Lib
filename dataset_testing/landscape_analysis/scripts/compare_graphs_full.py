import pandas as pd
import sys

def read_normal_file(name_file):

	rows = []

	file_open = open(name_file, 'r')

	line = file_open.readline()

	while line:

		line = line.replace("\n", "")
		rows.append(line)
		line = file_open.readline()

	file_open.close()

	return rows

original = pd.read_csv(sys.argv[1])
variant = pd.read_csv(sys.argv[2])

#read files in normal form
rows_original = read_normal_file(sys.argv[1])
rows_variante = read_normal_file(sys.argv[2])

rows_different_original = [row for row in rows_original if row not in rows_variante]
rows_different_variante = [row for row in rows_variante if row not in rows_original]

print("Rows different original v/s variant")
print(rows_different_original)

print("Rows different variant v/s original")
print(rows_different_variante)
