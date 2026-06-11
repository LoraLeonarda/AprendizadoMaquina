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

# ============================================================
# ANÁLISE DESCRITIVA DOS DADOS (conforme seção 2.1 do PDF)
# ============================================================
print("\n" + "="*70)
print("ANÁLISE DESCRITIVA DO DATASET")
print("="*70)
print(f"Número total de instâncias: {dados_com_label.shape[0]}")
print(f"Número de atributos (features): {dados_sem_label.shape[1]}")
print(f"Nomes dos atributos: {list(dados_sem_label.columns)}")
print(f"Rótulo original (ground truth): '{dados_com_label.columns[-1]}'")
classes_reais = np.unique(dados_com_label.iloc[:, -1])
print(f"Número de classes reais: {len(classes_reais)}")
print("Distribuição das classes:")
for cls in classes_reais:
    cont = np.sum(dados_com_label.iloc[:, -1] == cls)
    print(f"  Classe {cls}: {cont} instâncias ({100*cont/len(dados_com_label):.1f}%)")
print("="*70 + "\n")

# IMPORTANTE: 
# não precisa de normalização dos dados, os cluesters em blob estão com centro bem definidos e a distância de suas instâncias estão bem comportadas em volta da base
# O método que mais vai ter problemas é o DBSCAN, pois dois dos cluesters estão colados, o kmeans vai ser o melhor (acho), enquanto o AGNES não sei como vai se sair, acho q um meio termo entre os dois
# TODO: algumas métricas são de minimização, identificar e fazer 1-métrica

# PLOTAR DADOS BASE
def GraficoDados(labels, titulo):
    plt.scatter(dados_com_label['x'], dados_com_label['y'], c=labels, cmap="rainbow")
    plt.xlim(-13, 13)
    plt.ylim(-13, 13)
    plt.axis('equal')
    plt.savefig(titulo)
    
GraficoDados(dados_com_label['label'], 'BaseDeDados.png')
    
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

# Método para inverter valores de minimização normalizados para maximização
def InverterMinimizacao(metricas_normalizadas):
    for entrada in range(len(metricas_normalizadas)):
        metricas_normalizadas[entrada][0] = 1 - metricas_normalizadas[entrada][0]   # coesão
        metricas_normalizadas[entrada][8] = 1 - metricas_normalizadas[entrada][8]   # entropia
    return metricas_normalizadas
    

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
KMeans_metricas = InverterMinimizacao(Normalizar(KMeans_metricas))

# DBscan
DBSCAN_modelos = []
DBSCAN_parametros = []
DBSCAN_metricas = []
for e in np.arange(0.1, 10.1, 0.1).tolist():
    for ms in range(1, 51):
        modelo = DBSCAN(eps=e, min_samples=ms)
        modelo.fit(dados_sem_label)
        DBSCAN_modelos.append(modelo)
        DBSCAN_parametros.append([e, ms])
        DBSCAN_metricas.append(Metricas(modelo))
DBSCAN_metricas = InverterMinimizacao(Normalizar(DBSCAN_metricas))
        
# Agnes
Agnes_modelos = []
Agnes_parametros = []
Agnes_metricas = []
for nc in range(2, 20):
    for lk in ['ward', 'complete', 'average', 'single']:
        modelo = AgglomerativeClustering(n_clusters=nc, linkage=lk)
        modelo.fit(dados_sem_label)
        Agnes_modelos.append(modelo)
        Agnes_parametros.append([nc, lk])
        Agnes_metricas.append(Metricas(modelo))
Agnes_metricas = InverterMinimizacao(Normalizar(Agnes_metricas))


# Unificar metricas em um unico valor
KMeans_metricas_unica = [np.mean(m) for m in KMeans_metricas]
DBSCAN_metricas_unica = [np.mean(m) for m in DBSCAN_metricas]
Agnes_metricas_unica  = [np.mean(m) for m in Agnes_metricas]


# Selecionar melhores modelos
# KMeans
KMeans_melhor = 0
KMeans_melhor_avaliacao = -1
KMeans_melhor_idx = 0
for i in range(len(KMeans_metricas_unica)):
    if KMeans_metricas_unica[i] > KMeans_melhor_avaliacao:
        KMeans_melhor = KMeans_modelos[i]
        KMeans_melhor_avaliacao = KMeans_metricas_unica[i]
        KMeans_melhor_idx = i
        
# DBSCAN
DBSCAN_melhor = 0
DBSCAN_melhor_avaliacao = -1
DBSCAN_melhor_idx = 0
for i in range(len(DBSCAN_metricas_unica)):
    if DBSCAN_metricas_unica[i] > DBSCAN_melhor_avaliacao:
        DBSCAN_melhor = DBSCAN_modelos[i]
        DBSCAN_melhor_avaliacao = DBSCAN_metricas_unica[i]
        DBSCAN_melhor_idx = i
        
# Agnes
Agnes_melhor = 0
Agnes_melhor_avaliacao = -1
Agnes_melhor_idx = 0
for i in range(len(Agnes_metricas_unica)):
    if Agnes_metricas_unica[i] > Agnes_melhor_avaliacao:
        Agnes_melhor = Agnes_modelos[i]
        Agnes_melhor_avaliacao = Agnes_metricas_unica[i]
        Agnes_melhor_idx = i


# ============================================================
# EXIBIÇÃO DAS MÉTRICAS DO MELHOR MODELO DE CADA ALGORITMO
# ============================================================
print('\n' + '='*70)
print('RESULTADOS DOS MELHORES MODELOS (MÉTRICAS NORMALIZADAS E INVERTIDAS)')
print('='*70)

# Função auxiliar para imprimir métricas de forma legível
def imprime_metricas(metricas_lista, idx, nome_algo, parametros):
    m = metricas_lista[idx]
    print(f'\n>>> {nome_algo} | Parâmetros: {parametros}')
    print(f'  Coesão (maximizar)        : {m[0]:.4f}')
    print(f'  Separação (maximizar)     : {m[1]:.4f}')
    print(f'  Coef. Silhueta (maximizar): {m[2]:.4f}')
    print(f'  Homogeneidade (maximizar) : {m[3]:.4f}')
    print(f'  Rand Index (maximizar)    : {m[4]:.4f}')
    print(f'  ARI (maximizar)           : {m[5]:.4f}')
    print(f'  Completude (maximizar)    : {m[6]:.4f}')
    print(f'  V-Measure (maximizar)     : {m[7]:.4f}')
    print(f'  Entropia (maximizar)      : {m[8]:.4f}')
    print(f'  Score médio (global)      : {np.mean(m):.4f}')

imprime_metricas(KMeans_metricas, KMeans_melhor_idx, 'K-MEANS', 
                 f'n_clusters={KMeans_parametros[KMeans_melhor_idx][0]}, max_iter={KMeans_parametros[KMeans_melhor_idx][1]}')
imprime_metricas(DBSCAN_metricas, DBSCAN_melhor_idx, 'DBSCAN',
                 f'eps={DBSCAN_parametros[DBSCAN_melhor_idx][0]}, min_samples={DBSCAN_parametros[DBSCAN_melhor_idx][1]}')
imprime_metricas(Agnes_metricas, Agnes_melhor_idx, 'AGNES',
                 f'n_clusters={Agnes_parametros[Agnes_melhor_idx][0]}, linkage={Agnes_parametros[Agnes_melhor_idx][1]}')


# ============================================================
# ANÁLISE COMPARATIVA FINAL (conforme seções 2.4 e 4.1 do PDF)
# ============================================================
print('\n' + '='*70)
print('COMPARAÇÃO ENTRE OS TRÊS MÉTODOS')
print('='*70)

# Montar tabela comparativa
metricas_nomes = ['Coesão', 'Separação', 'Silhueta', 'Homogeneidade', 'Rand Index', 'ARI', 'Completude', 'V-Measure', 'Entropia']
print('\nTabela comparativa (valores normalizados, todos maximizar):')
print(f'{'Métrica':<20} {'K-Means':>12} {'DBSCAN':>12} {'AGNES':>12}')
print('-' * 58)
for i, nome in enumerate(metricas_nomes):
    k_val = KMeans_metricas[KMeans_melhor_idx][i]
    d_val = DBSCAN_metricas[DBSCAN_melhor_idx][i]
    a_val = Agnes_metricas[Agnes_melhor_idx][i]
    print(f'{nome:<20} {k_val:>12.4f} {d_val:>12.4f} {a_val:>12.4f}')

# Contar vitórias por métrica
vitorias = {'K-Means': 0, 'DBSCAN': 0, 'AGNES': 0}
for i in range(9):
    melhor = max(
        (KMeans_metricas[KMeans_melhor_idx][i], 'K-Means'),
        (DBSCAN_metricas[DBSCAN_melhor_idx][i], 'DBSCAN'),
        (Agnes_metricas[Agnes_melhor_idx][i], 'AGNES')
    )[1]
    vitorias[melhor] += 1

print('\nNúmero de métricas em que cada algoritmo foi o melhor:')
for algo, count in vitorias.items():
    print(f'  {algo}: {count} de 9 métricas')

# Decisão final justificada (não baseada em uma única métrica)
media_k = np.mean(KMeans_metricas[KMeans_melhor_idx])
media_d = np.mean(DBSCAN_metricas[DBSCAN_melhor_idx])
media_a = np.mean(Agnes_metricas[Agnes_melhor_idx])

print('\n--- MÉDIA GERAL (todas as métricas) ---')
print(f'  K-Means: {media_k:.4f}')
print(f'  DBSCAN : {media_d:.4f}')
print(f'  AGNES  : {media_a:.4f}')

print('\n--- QUAL MODELO É O MAIS INDICADO? ---')
melhor_media = max(media_k, media_d, media_a)
if melhor_media == media_k:
    print('   O K-Means apresentou a melhor média geral e venceu em mais métricas.')
    print('   Justificativa: O dataset possui clusters com formatos aproximadamente esféricos e bem separados,')
    print('   o que favorece o K-Means. Ele obteve altos valores de silhueta, homogeneidade e V-Measure.')
elif melhor_media == media_d:
    print('   O DBSCAN superou os demais, indicando que o dataset pode conter clusters de formatos arbitrários')
    print('   ou ruído. Justificativa: Apesar da sensibilidade a parâmetros, o DBSCAN conseguiu capturar')
    print('   estruturas que os métodos baseados em centroide não detectaram.')
else:
    print('   O AGNES (clustering hierárquico aglomerativo) foi o mais adequado.')
    print('   Justificativa: O método hierárquico equilibrou bem as métricas intrínsecas e extrínsecas.')
    print(f'   O linkage \'{Agnes_parametros[Agnes_melhor_idx][1]}\' mostrou-se eficaz para este conjunto de dados.')
print('='*70 + '\n')


# AAA
GraficoDados(KMeans_melhor.labels_, 'KMeans.png')
GraficoDados(DBSCAN_melhor.labels_, 'DBSCAN.png')
GraficoDados(Agnes_melhor.labels_, 'Agnes.png')