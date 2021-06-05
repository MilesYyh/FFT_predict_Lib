Explicación componentes generales y ejecución de scripts

extraction_data_from_landscape.py : permite extraer la información del landscape y habilitarla en un único archivo csv, posee declarado un path donde fueron almacenados los resultados desde el servicio SDM.

make_landscape.py : genera un archivo txt con las mutaciones simuladas para todas las posiciones

run_sdm_service.py : corre el servicio sdm, ejecutando el script sdm_service, importante mencionar que posee declarado un path con los archivos txt generados por el splitter_files script.

sdm_service.py : ejecuta el servicio SDM consumiendo la app de manera automática. Tiene declarado el archivo PDB a utilizar y también el path donde se dejará los resultados. Además, recibe como input el archivo txt con las mutaciones.

splitter_files.py : Divide el archivo landscape generado en archivos de tamaño 20 debido a los requerimientos del servicio SDM.

Pipeline de ejecución

Para obtener los resultados de manera satisfactoria, se debe ejecutar de la siguiente manera:

1. make_landscape.py
2. splitter_files.py
3. run_sdm_service.py
4. extraction_data_from_landscape.py

Recordar que sdm_service.py es utilizado por el script run_sdm_service.py.

NOTA: en el caso de que algún componente del run_sdm falle, esto es, alguna iteración no la procesa de manera correcta, se recomienda ejecutar el proceso a mano para dicha iteración, o reintentar modificando el script una vez finalice, alterando la lista del for.

