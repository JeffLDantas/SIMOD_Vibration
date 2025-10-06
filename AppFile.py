from pathlib import Path
from Manobra import Manobra
import os
import sys
from Sinal import Sinal
from enums import CircuitBreaker
from enums import Actuation
import numpy as np
from vmdpy import VMD
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import matplotlib.pyplot as plt

class AppFile():
    def __init__(self, circuitBreaker: CircuitBreaker, actuation: Actuation, tag):
        self.sinaisManobra={}
        self.circuitBreaker = circuitBreaker
        self.actuation = actuation
        self.tag = tag

    def lerArquivo(self,sinalTag,pasta):
        try:
            tempos=[]
            dados=[]
            inf={}

            nomeArquivo=f'{pasta}\\{sinalTag}.csv'
            caminho_arquivo = Path(nomeArquivo)
            if caminho_arquivo.exists():
                #print(f'Lendo Sinal {sinalTag}')
                with open(nomeArquivo, 'r') as arquivo:
                    # Ler o texto no arquivo
                    c=0
                    for linha in arquivo:
                        # Processar cada linha
                        valores = linha.strip().split(',')
                        
                        if c>3:
                            tempos.append(float(valores[0]))
                            dados.append(float(valores[1]))
                        elif c==1:
                            inf["tag"]=valores[0].lstrip()
                            inf["descricao"]=valores[1].lstrip()
                            inf["modulo"]=int(valores[2])
                            inf["fase"]=valores[3].lstrip()
                            inf["grandeza"]=valores[4].lstrip()
                            inf["idSensor"]=valores[5].lstrip()
                            inf["frequenciaAmostragem"]=float(valores[6])
                            inf["timestamp"]=float(valores[7])
                        c+=1
                    return 1,Sinal(inf,dados,tempos)
            else:
                #print(f'Arquivo {nomeArquivo} não existe!')
                return 0, 0
        except Exception as e:
                print('Ocorreu um erro durante a leitura dos arquivos, verifique a lista de tags e o diretório com as manobra:', e)
                sys.exit('Sistema Finalizado!')


    def obterSinaisManobra(self, path, sinalTag):
        sinaisManobra={}
        caminho= Path(path)
        dados = []
        tempos = []
        samplingRate = 0
        if caminho.exists():
            resp,s=self.lerArquivo(sinalTag,caminho)
            if resp==1:
                sinaisManobra[sinalTag]=s
                #print(f'Sinal {sinalTag} da manobra {valores[1]} lido com sucesso.')
        
        if len(sinaisManobra) > 0:
            dados = sinaisManobra[sinalTag].getDados()
            tempos = sinaisManobra[sinalTag].getTempos()
            date = sinaisManobra[sinalTag].getDataHora()
            if(sinaisManobra[sinalTag].frequencia > 0):
                samplingRate = sinaisManobra[sinalTag].frequencia

        return dados, samplingRate, tempos, date
    
    def GetAllSignals(self):
        manobra = f'N'
        if self.actuation == Actuation.Opening:
            manobra = f'A'
        elif self.actuation == Actuation.Closing:
            manobra = f'F'
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        diretorio = Path(diretorio_atual+f'\\manobras\\{self.circuitBreaker.name}')
        subfolders = [ f.path for f in os.scandir(diretorio) if f.is_dir() and manobra in f.name ]
        signals = []
        events = []
        eventsWithDate = []
        for folder in subfolders:
            dados, frequencia, tempos, date = self.obterSinaisManobra(folder, self.tag)
            if len(dados) > 0:
                dados = np.array(dados)
                signals.append(dados)
                event = ''.join(filter(str.isdigit, folder))
                events.append(f'{event}')
                eventsWithDate.append(f'{event} - {manobra} - {date}')
            if frequencia > 0:
                samplingRate = frequencia

        minSize = min(len(arr) for arr in signals)

        # i = 0
        # for signal in signals:
        #     if i == 10:
        #         signals[i] = signal[24040:len(signal)]
        #         signal = signals[i]
        #         signals[i] = signal[:minSize]
        #         self.PrintCurve(signals[i], events[i])
        #     i = i + 1
        
        signals = [arr[(len(arr) - minSize):len(arr)] for arr in signals]

        signals = np.array(signals)

        events = np.array(events)

        # for i in range(len(signals)):
        #     self.PrintCurve(signals[i], events[i])

        return signals, events, samplingRate, minSize, eventsWithDate
    
    def ApplyingVMD(self, dados, modes, sampleSize, signalNumber):
        # Apply VMD
        alpha = 9000                # Bandwidth constraint
        tau = 1/sampleSize          # Time-step of the decomposition
        K = modes                   # Number of modes to decompose into
        DC = 0                      # DC component inclusion
        init = 1                    # Initialize the modes
        tol = 1e-7                  # Tolerance for convergence

        # VMD decomposition
        print(f'Aplicando VMD! Signal: {signalNumber}.')
        u, u_hat, omega = VMD(dados, alpha, tau, K, DC, init, tol)
        return u, u_hat, omega
    
    def GetAllIMFsFromVMD(self):
        signals, events, samplingRate, sampleSize, eventsWithDate = self.GetAllSignals()
        imfs = []
        imfsFrequencyDomain = []
        modes = 7
        signalSize = len(signals)
        print(f'Signal quantity: {signalSize}.')
        for i in range(signalSize):
            u, u_hat, omega = self.ApplyingVMD(signals[i], modes, sampleSize, i + 1)
            imfs.append(u)
            imfsFrequencyDomain.append(u_hat)
            n_samples = len(u_hat)
        
        imfs = np.array(imfs)
        imfsFrequencyDomain = np.array(imfsFrequencyDomain)

        return imfs, samplingRate, imfsFrequencyDomain, events, modes, n_samples, eventsWithDate
    
    def GetExtractFeaturesFromVMD(self, imfs, samplingRate):
        features = []
        for imf in imfs:
            # Time-domain features
            rms = np.sqrt(np.mean(imf**2))
            pk2pk = np.max(imf) - np.min(imf)
            kur = kurtosis(imf)
            sk = skew(imf)
        
            # Frequency-domain features
            freqs, psd = welch(imf, fs=samplingRate) 
            dominant_freq = freqs[np.argmax(psd)]
            spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))
        
            features.extend([rms, pk2pk, kur, sk, dominant_freq, spectral_entropy])
        return np.array(features)
    
    def GetExtractFeaturesFromVMDDomainFrequency(self, imfs, samplingRate, modes, n_samples):
        features = []
        f = np.arange(samplingRate/-2, samplingRate/2, samplingRate/n_samples)
        half_samples = int(n_samples/2)
        f = f[half_samples:n_samples]
 
        for i in range(modes):
            absolutValues = np.abs(imfs[half_samples:n_samples, i]) + 1e-10  # avoid log(0)
            centroid = np.sum(f * absolutValues) / np.sum(absolutValues)
            bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * absolutValues) / np.sum(absolutValues))
            flatness = np.exp(np.mean(np.log(absolutValues))) / np.mean(absolutValues)
            features.extend([centroid, bandwidth, flatness])
            
        return np.array(features)

    def GetAllExtractedFeaturesFromVMD(self):
        allIMFs, samplingRate, allIMFsFrequencyDomain, events, modes, n_samples = self.GetAllIMFsFromVMD()
        allFeatures = np.array([self.GetExtractFeaturesFromVMD(imfs, samplingRate) for imfs in allIMFs])
        allFrequencyDomainFeatures = np.array([self.GetExtractFeaturesFromVMDDomainFrequency(imfsFD, samplingRate, modes, n_samples) for imfsFD in allIMFsFrequencyDomain])
        return allFeatures, allFrequencyDomainFeatures, events
    
    def PrintCurve(self, dado, event):
        x = np.arange(len(dado))
        plt.figure(figsize=(10, 4))
        plt.plot(x, dado, label=event)
        plt.legend()
        plt.tight_layout()
        plt.show()