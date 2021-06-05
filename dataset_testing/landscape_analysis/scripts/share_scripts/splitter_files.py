import urllib
from bs4 import BeautifulSoup
import re
import urllib.parse as urlparse
from urllib.parse import parse_qs
import sys  
import mechanize
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Mostrar información de depuración", action="store_true")
parser.add_argument("-m", "--mutationFile", help="Nombre de archivo de mutaciones a procesar")
parser.add_argument("-p", "--pdb", help="Nombre de archivo pdb a procesar")
parser.add_argument("-r", "--result", help="Lugar para almacenar resultados")

args = parser.parse_args()
 
# Aquí procesamos lo que se tiene que hacer con cada argumento
if args.verbose and args.verbose=='v':
    print ("depuración activada!!!")
 
if args.mutationFile and args.verbose=='v':
    print ("El nombre de archivo de mutaciones a procesar es: ", args.mutationFile)
if args.pdb and args.verbose=='v':
    print ("El nombre de archivo a procesar es: ", args.pdb)
if args.result and args.verbose=='v':
    print ("Lugar para almacenar resultados: ", args.result)


mutationFile = args.mutationFile
pdb=args.pdb
path = args.result

#print ("Start : %s" % time.ctime())

#https://stackoverflow.com/questions/16289859/splitting-large-text-file-into-smaller-text-files-by-line-numbers-using-python
#divido archivo de mutaciones 
#en el maximo aceptado
lines_per_file = 20
smallfile = None
listMutationFile = []
with open(mutationFile) as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'small_file_{}.txt'.format(lineno + lines_per_file)
            listMutationFile.append(small_filename)
            smallfile = open(path+small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()

'''
#envia uno a uno los job 
#es posible que el servidor del sdm detecta trabajo multiple
#y bloquee ip desde donde lleguen los trabajos
# se debe colocar una pausa
# se debe paralelizar trabajo

for x in range(0,len(listMutationFile)):
    # accedo a url principal
    url = "http://marid.bioc.cam.ac.uk/sdm2/prediction"
    br = mechanize.Browser()
    br.open(url)
    
    #obtengo formulario para mutaciones
    br.select_form(nr=2)
    
    #completo formulario de pagina
    br.form.add_file(open(pdb), 'text/plain', pdb, name='wild')
    br.form.add_file(open(path+listMutationFile[x]), 'text/plain', path+listMutationFile[x], name='mutation_list')
    
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
    #vuelta = 0
    
    #ciclo solo se rompe si no encuentra la palabra running
    # si la encuentra devuelve posicion en el texto que ocupa
    # si no la encuentra devuelve -1 y rompe ciclo
    while(bandera>0):
        response = br.open(pathResp)
        #si detecto aun un hilo del job en estado running sigo esperando
        bandera = response.read().decode().find("Running")
        #tiempo de refresco de la pagina
        time.sleep( 10 )
        #vuelta = vuelta + 1 
        #print (str(vuelta))
        #print (str(bandera))

    
    #Si ciclo anterior termina recupera tar con el resultado
    pathSDM = "http://marid.bioc.cam.ac.uk/sdm2/static/results/"+marca+"_download.tar"
    #recuperar archivo
    response = br.open(pathSDM)
    #guardar archivo
    f = open(path+marca+"_download.tar", "wb")
    f.write(response.read())
    
    #print (marca+" has been downloaded")

#print ("End : %s" % time.ctime())
'''