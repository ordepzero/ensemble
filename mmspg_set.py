# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:03:12 2018

@author: PeDeNRiQue
"""
import sample
import scipy.io as scio

# Arquivos do Subject1 Sessao1 
MMSPG_SUB1SES1ARQ1 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191428_epochs.mat"
MMSPG_SUB1SES1ARQ2 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191430_epochs.mat"
MMSPG_SUB1SES1ARQ3 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191431_epochs.mat"
MMSPG_SUB1SES1ARQ4 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191433_epochs.mat"
MMSPG_SUB1SES1ARQ5 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191435_epochs.mat"
MMSPG_SUB1SES1ARQ6 = "../../bases/mmspg/subject1/subject1/session1/eeg_200605191437_epochs.mat"
MATLAB_FILE = MMSPG_SUB1SES1ARQ1 # MATLAB_FILE  é o arquivo que será analizado
ARRAY_TO_TIME = 1366 # 1366 posicoes no array para 0.667 segundos (considerando 2048Hz)
STIMULUS_FREQ = 820 # o primeiro stimulo é gerado a 0.4seg assim como os estimulos seguintes são a cada 0.4 seg


# Esta funcao verifica se o sinal que piscou era o alvo
# que o individuo estava observando
def has_p300(stimuli, target):
    answer = []
    for x in stimuli:
        answer.append(1 if x == target else 0)
    return answer

# Esta funcao separa o sinal em uma matriz de samples com suas respectivas respostas 
def generate_sample_matrix(data, answer):
    sample_matrix = []
    for idx, canal in enumerate(data):
        sample_matrix.append([])
        for i in range(len(answer)):
            #soma-se +1 no primeiro pois o sinal começa na posicao STIMULUS_FREQ
            sampl = canal[(i+1)*STIMULUS_FREQ:(i+1)*STIMULUS_FREQ + ARRAY_TO_TIME] 
            sample_matrix[idx].append([])
            sample_matrix[idx][i].append(sampl)
            sample_matrix[idx][i].append(answer[i])
            
    return sample_matrix


def transform_into_samples(mmspg_set):
    
    n_channels = len(mmspg_set)#34
    n_samples = len(sample_matrix[0])#138
    samples = []
    for i in range(n_samples):
        samples_signals = []
        for j in range(n_channels):
            samples_signals.append(sample_matrix[j][i][0])
            
        samples.append(sample.Sample(samples_signals,sample_matrix[j][i][1]))
        #sample_matrix[j][i]
            
    return samples

def load_set(conf):
    mmspg_set = scio.loadmat(MATLAB_FILE)

#    Eletroencefalogramas    
    mmspg_set['data']
#   Qual imagem estava piscando
    mmspg_set['stimuli']
#   Qual valor o individuo estava olhando
    mmspg_set['target']
#   Qual valor o individuo estava olhando
    mmspg_set['targets_counted']

#    Events possui a data e a hora que ocorreu o evento
#    Sendo que o primeiro evento ocorreu 0.4 segundos
#    após o inicio do experimento
    mmspg_set['events']
#   List com as respostas
    answer = has_p300(mmspg_set['stimuli'][0], mmspg_set['target'])

#   sample_matrix[0:33] = canal
#   sample_matrix[0:33][0:138] = samples
#   sample_matrix[0:33][0:138][0][0:ARRAY_TO_TIME] = sinais
#   sample_matrix[0:33][0:138][1] = 0 sem p300, 1 com p300
    sample_matrix = generate_sample_matrix(mmspg_set['data'], answer)
    
    return sample_matrix

if __name__ == '__main__': 
    
    mmspg_set = scio.loadmat(MATLAB_FILE)

#    Eletroencefalogramas    
    mmspg_set['data']
#   Qual imagem estava piscando
    mmspg_set['stimuli']
#   Qual valor o individuo estava olhando
    mmspg_set['target']
#   Qual valor o individuo estava olhando
    mmspg_set['targets_counted']

#    Events possui a data e a hora que ocorreu o evento
#    Sendo que o primeiro evento ocorreu 0.4 segundos
#    após o inicio do experimento
    mmspg_set['events']
#   List com as respostas
    answer = has_p300(mmspg_set['stimuli'][0], mmspg_set['target'])

#   sample_matrix[0:33] = canal
#   sample_matrix[0:33][0:138] = samples
#   sample_matrix[0:33][0:138][0][0:ARRAY_TO_TIME] = sinais
#   sample_matrix[0:33][0:138][1] = 0 sem p300, 1 com p300
    sample_matrix = generate_sample_matrix(mmspg_set['data'], answer)
    
    #all_samlpes = transform_into_samples(sample_matrix)