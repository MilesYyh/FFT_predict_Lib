'''
clase que permite estimar los estadisticos de un data set con respecto a la informacion que este posee,
recibe como atributos el data set...
'''

import numpy as np
import pandas as pd

class createStatisticSummary(object):

    def __init__(self, dataSet):

        self.dataSet = pd.read_csv(dataSet)

    #metodo que permite calcular los estadisticos para una columna en el set de datos...
    def calculateValuesForColumn(self, attribute):

        dictResponse = {}
        dictResponse.update({'mean': np.mean(self.dataSet[attribute])})
        dictResponse.update({'std': np.std(self.dataSet[attribute])})
        dictResponse.update({'var': np.var(self.dataSet[attribute])})
        dictResponse.update({'max': max(self.dataSet[attribute])})
        dictResponse.update({'min': min(self.dataSet[attribute])})

        return dictResponse
