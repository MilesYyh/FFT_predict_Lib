import urllib
from bs4 import BeautifulSoup
import re
import urllib.parse as urlparse
from urllib.parse import parse_qs
import sys  
import mechanize
import time
import os

pdb = "1rx4.pdb"
input_file = sys.argv[1]
results = "result_landscape/"

name_dir = input_file.split(".")[0]

command = "mkdir -p %s%s" % (results, name_dir) 
os.system(command)

# accedo a url principal
url = "http://marid.bioc.cam.ac.uk/sdm2/prediction"

br = mechanize.Browser()
br.open(url)

#obtengo formulario para mutaciones
br.select_form(nr=2)
    
#completo formulario de pagina
br.form.add_file(open(pdb), 'text/plain', pdb, name='wild')
br.form.add_file(open(input_file), 'text/plain', input_file, name='mutation_list')
    

#genero sumbit sobre formulario de pagina
response = br.submit()
    
#obtengo respuesta de pagina con codigo para revisar
form_result = response.read()

# obtengo url con codigo de job
pathResp = br.geturl()
parsed = urlparse.urlparse(pathResp)

#obtengo marca del job
marca = parse_qs(parsed.query)['job_id'][0]

#preguntar si trabajo esta terminado
bandera = 1

    
#ciclo solo se rompe si no encuentra la palabra running
# si la encuentra devuelve posicion en el texto que ocupa
# si no la encuentra devuelve -1 y rompe ciclo
iteration=0
while(bandera>0):

	response = br.open(pathResp)
    #si detecto aun un hilo del job en estado running sigo esperando
	bandera = response.read().decode().find("Running")
    #tiempo de refresco de la pagina
	time.sleep( 10 )
	print("iteration: ", iteration)
	iteration+=1


#Si ciclo anterior termina recupera tar con el resultado
pathSDM = "http://marid.bioc.cam.ac.uk/sdm2/static/results/"+marca+"_download.tar"

#recuperar archivo
response = br.open(pathSDM)
#guardar archivo
f = open(results+name_dir+"/"+marca+"_download.tar", "wb")

f.write(response.read())
