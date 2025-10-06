
import matplotlib.pyplot as plt
from datetime import datetime
'''
Classe Sinal utilizada para armazenar os sinais lidos de uma manobra.
O tempo armazenado é em milissegundos.
O timestamp é em milissegundos. 
'''
class Sinal:
    def __init__(self,inf,dados, tempos):
        #dados do sinal
        self.tag= inf["tag"]
        self.descricao=inf["descricao"]
        self.modulo=inf["modulo"]
        self.fase=inf["fase"]
        self.grandeza=inf["grandeza"]
        self.idSensor=inf["idSensor"]
        self.frequencia=inf["frequenciaAmostragem"]
        self.timestamp=inf["timestamp"]
        self.dados=dados
        self.tempos=tempos

    def getTag(self):
        return self.tag

    def getDescricao(self):
        return self.descricao
    
    def getModulo(self):
        return self.modulo
    
    def getFase(self):
        return self.fase
    
    def getGrandeza(self):
        return self.grandeza
    
    def getIdSensor(self):
        return self.idSensor
    
    def getFrequencia(self):
        return self.frequencia
    
    def getTimestamp(self):
        return self.timestamp

    def getDados(self):
        return self.dados
    
    def getTempos(self):
        return self.tempos
              
    def plotaSinal(self):
        plt.figure(figsize=(20, 10))
        plt.xlabel('Tempo (ms)')
        if self.grandeza=='I':
            plt.ylabel('Amperes')
        elif self.grandeza=='V':
            plt.ylabel('Volts')
        else:
            plt.ylabel('Valores')

        plt.title(f'Gráfico Sinal - {self.tag} - {self.descricao}')
        plt.plot(self.tempos, self.dados, label=self.tag)        
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def getDataHora(self):
        aux= datetime.fromtimestamp(self.timestamp)
        # Extraindo e exibindo partes individuais
        return f'{aux.day}/{aux.month}/{aux.year} - {aux.hour}:{aux.minute}:{aux.second}'
    

   
