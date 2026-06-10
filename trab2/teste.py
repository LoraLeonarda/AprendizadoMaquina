# IMPORT UTILS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# IMPORT MODELOS
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# IMPORT METRICAS
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, rand_score
from scipy.stats import entropy

# ABRIR DATASET
dataset = pd.read_csv('dataset.csv')
dados_com_label = pd.DataFrame(dataset)
dados_sem_label = pd.DataFrame(dataset,columns=dataset.columns[:-1])

# IMPORTANTE: 
# não precisa de normalização dos dados, os cluesters em blob estão com centro bem definidos e a distância de suas instâncias estão bem comportadas em volta da base
# O método que mais vai ter problemas é o DBSCAN, pois dois dos cluesters estão colados, o kmeans vai ser o melhor (acho), enquanto o AGNES não sei como vai se sair, acho q um meio termo entre os dois
# TODO: algumas métricas são de minimização, identificar e fazer 1-métrica

# PLOTAR DADOS BASE
def GraficoDados(labels):
    plt.scatter(dados_com_label['x'], dados_com_label['y'], c=labels, cmap="rainbow")
    plt.xlim(-13, 13)
    plt.ylim(-13, 13)
    plt.axis('equal')
    plt.show()
    
GraficoDados(dados_com_label['label'])
    
# FUNÇÃO PARA CALCULAR MÉTRICAS
def Metricas(modelo):
    
    # Variaveis
    X = dados_sem_label.values
    n_total = len(X)
    y_true = dados_com_label.iloc[:, -1].values
    y_pred = modelo.labels_
    unique_clusters = np.unique(y_pred)
    n_clusters = len(unique_clusters)
    
    centroides = []
    for cluster in unique_clusters:
        if cluster == -1:
            continue
        pontos_cluster = X[y_pred == cluster]
        if len(pontos_cluster) == 0:
            continue
        centroides.append(np.mean(pontos_cluster, axis=0))
    
    
    # metodos manuais
    # separação
    separacao = 0.0
    for i in range(len(centroides)):
        for j in range(i + 1, len(centroides)):
            separacao += np.linalg.norm(
                centroides[i] - centroides[j]
            )
            
            
    # entropia
    entropia = 0.0
    for cluster in unique_clusters:
        if cluster == -1:
            continue
        indices = np.where(y_pred == cluster)[0]
        if len(indices) == 0:
            continue
        classes = y_true[indices]
        _, contagens = np.unique(
            classes,
            return_counts=True
        )
        probs = contagens / contagens.sum()
        entropia_cluster = entropy(probs, base=2)
        entropia += (
            len(indices) / n_total
        ) * entropia_cluster
    
    
    # coesão
    coesao = 0.0
    for cluster in unique_clusters:
        # ignora ruído do DBSCAN
        if cluster == -1:
            continue
        pontos_cluster = X[y_pred == cluster]
        if len(pontos_cluster) == 0:
            continue
        centroide = np.mean(pontos_cluster, axis=0)
        distancias = np.linalg.norm(
            pontos_cluster - centroide,
            axis=1
        )
        coesao += np.sum(distancias)

    # Silhueta com check especial
    if n_clusters >= 2 and len(X) > n_clusters and len(set(y_pred)) > 1:
        silhueta = (silhouette_score(X, y_pred, metric='euclidean') + 1) / 2.0
    else:
        silhueta = 0.0

    # Métricas extrinsicas
    homogeneidade  = homogeneity_score(y_true, y_pred)
    completude     = completeness_score(y_true, y_pred)
    v_measure      = v_measure_score(y_true, y_pred)
    rand_score_val = rand_score(y_true, y_pred)
    ari            = (adjusted_rand_score(y_true, y_pred) + 1) / 2.0

    # Vetor de métricas (com pesos)
    metricas = []
    metricas.append(coesao)
    metricas.append(separacao)
    metricas.append(silhueta)
    metricas.append(homogeneidade)
    metricas.append(rand_score_val)
    metricas.append(ari)
    metricas.append(completude)
    metricas.append(v_measure)
    metricas.append(entropia)

    # retorno de métricas
    return metricas
    
# Método para normalizar vetor de métricas passado
def Normalizar(metricas):
    
    # Identificar ranges
    maiores = []
    menores = []
    
    # Para cada metrica
    for metrica in range(len(metricas[0])):
        maiores.append(max([dado[metrica] for dado in metricas]))
        menores.append(min([dado[metrica] for dado in metricas]))
        
    # Aplicar transformações ((valor - menor) / (maior - menor))
    # Para cada entrada
    for entrada in range(len(metricas)):
        # Para cada metrica
        for metrica in range(len(metricas[entrada])):
            metricas[entrada][metrica] = (metricas[entrada][metrica]-menores[metrica]) / (maiores[metrica] - menores[metrica])
    
    # Retorna
    return metricas

# KMeans
KMeans_modelos = []
KMeans_parametros = []
KMeans_metricas = []
for nc in range(2, 7):
    for mi in [5, 10, 20, 40, 100, 200]:
        modelo = KMeans(n_clusters=nc, max_iter = mi, random_state = 77)
        modelo.fit(dados_sem_label)
        KMeans_modelos.append(modelo)
        KMeans_parametros.append([nc, mi])
        KMeans_metricas.append(Metricas(modelo))
KMeans_metricas = Normalizar(KMeans_metricas)

# DBscan
DBSCAN_modelos = []
DBSCAN_parametros = []
DBSCAN_metricas = []
for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 2, 2.5, 3]:
    for ms in range(2, 51):
        modelo = DBSCAN(eps=e, min_samples=ms)
        modelo.fit(dados_sem_label)
        DBSCAN_modelos.append(modelo)
        DBSCAN_parametros.append([e, ms])
        DBSCAN_metricas.append(Metricas(modelo))
DBSCAN_metricas = Normalizar(DBSCAN_metricas)
        
# Agnes
Agnes_modelos = []
Agnes_parametros = []
Agnes_metricas = []
for nc in range(2, 7):
    for lk in ['ward', 'complete', 'average', 'single']:
        modelo = AgglomerativeClustering(n_clusters=nc, linkage=lk)
        modelo.fit(dados_sem_label)
        Agnes_modelos.append(modelo)
        Agnes_parametros.append([nc, lk])
        Agnes_metricas.append(Metricas(modelo))
Agnes_metricas = Normalizar(Agnes_metricas)


# Unificar metricas em um unico valor
KMeans_metricas_unica = [np.mean(m) for m in KMeans_metricas]
DBSCAN_metricas_unica = [np.mean(m) for m in DBSCAN_metricas]
Agnes_metricas_unica  = [np.mean(m) for m in Agnes_metricas]


# Selecionar melhores modelos
# KMeans
KMeans_melhor = 0
KMeans_melhor_avaliacao = -1
for i in range(len(KMeans_metricas_unica)):
    if KMeans_metricas_unica[i] > KMeans_melhor_avaliacao:
        KMeans_melhor = KMeans_modelos[i]
        KMeans_melhor_avaliacao = KMeans_metricas_unica[i]
        
# DBSCAN
DBSCAN_melhor = 0
DBSCAN_melhor_avaliacao = -1
for i in range(len(DBSCAN_metricas_unica)):
    if DBSCAN_metricas_unica[i] > DBSCAN_melhor_avaliacao:
        DBSCAN_melhor = DBSCAN_modelos[i]
        DBSCAN_melhor_avaliacao = DBSCAN_metricas_unica[i]
        
# Agnes
Agnes_melhor = 0
Agnes_melhor_avaliacao = -1
for i in range(len(Agnes_metricas_unica)):
    if Agnes_metricas_unica[i] > Agnes_melhor_avaliacao:
        Agnes_melhor = Agnes_modelos[i]
        Agnes_melhor_avaliacao = Agnes_metricas_unica[i]


# AAA
GraficoDados(KMeans_melhor.labels_)
GraficoDados(DBSCAN_melhor.labels_)
GraficoDados(Agnes_melhor.labels_)
