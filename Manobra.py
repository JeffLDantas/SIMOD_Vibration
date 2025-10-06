from datetime import datetime
'''
Classe Manobra utilizada para armazenar os dados da Manobra.
O tempo armazenado é em milissegundos.
O timestamp é em milissegundos. 
'''
class Manobra:
    def __init__(self,inf):
        #dados do sinal
        self.idDisjuntor= inf["idDisjuntor"]
        self.idEvento=inf["idEvento"]
        self.idManobra=inf["idManobra"]
        self.tipo=inf["tipo"]
        self.timeStamp=inf["timeStamp"]/1000
        self.inicioA=inf["inicioA"]
        self.inicioB=inf["inicioB"]
        self.inicioC=inf["inicioC"]
        self.fimA=inf["fimA"]
        self.fimB=inf["fimB"]
        self.fimC=inf["fimC"]
        

    def getIdDisjuntor(self):
        return self.idDisjuntor

    def getIdEvento(self):
        return self.idEvento
    
    def getidManobra(self):
        return self.idManobra
    
    def getTipo(self):
        return self.tipo
    
    def getTimeStamp(self):
        return self.timeStamp
    
    def getInicioA(self):
        return self.inicioA
    
    def getInicioB(self):
        return self.inicioB
    
    def getInicioC(self):
        return self.inicioC
    
    def getFimA(self):
        return self.fimA
    
    def getFimB(self):
        return self.fimB
    
    def getFimC(self):
        return self.fimC
    
    def getDataHora(self):
        aux= datetime.fromtimestamp(self.timeStamp)
        # Extraindo e exibindo partes individuais
        return f'{aux.day}/{aux.month}/{aux.year} - {aux.hour}:{aux.minute}:{aux.second}'
    
    
   
