# Bibliotecas basicas
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Métodos de split e métricas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# ------------------------------
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ------------------------------
dataset = pd.read_csv('ShoppingDataset.csv')

# Mapeamentos
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
city_tier_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
shopping_pref_map = {'Store': 0, 'Online': 1, 'Hybrid': 2}

dataset['gender'] = dataset['gender'].map(gender_map)
dataset['city_tier'] = dataset['city_tier'].map(city_tier_map)
dataset['shopping_preference'] = dataset['shopping_preference'].map(shopping_pref_map)

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

# Divisão estratificada: 50% treino, 25% validação, 25% teste
x_treino, x_temp, y_treino, y_temp = train_test_split(
    X, Y, test_size=0.5, stratify=Y, random_state=42
)
x_validacao, x_teste, y_validacao, y_teste = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ------------------------------
# AVALIAÇÃO COM PARÂMETROS DEFAULT
# ------------------------------
modelos_default = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(),
    "Bagging": BaggingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

print("=== Acurácia com parâmetros default ===\n")
for nome, modelo in modelos_default.items():
    modelo.fit(x_treino, y_treino)
    opiniao = modelo.predict(x_teste)
    acc = accuracy_score(y_teste, opiniao)
    f1 = f1_score(y_teste, opiniao, average='weighted')
    print(f"{nome:15} | Acurácia: {acc:.4f} | F1-score: {f1:.4f}")


# ------------------------------
# BUSCA DE HIPERPARÂMETROS
# ------------------------------

# KNN
print("Treinando KNN...")
KNN_melhor = KNeighborsClassifier()
KNN_maior = -1
KNN_params = []

for wei in ['uniform', 'distance']:
    for nei in [3, 5, 7, 9, 11, 13, 15]:
        print("\tTreinando ", [wei, nei])
        KNN = KNeighborsClassifier(n_neighbors = nei, weights = wei)
        KNN.fit(x_treino, y_treino)
        opiniao = KNN.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        if (Acc > KNN_maior):
            KNN_maior = Acc
            KNN_melhor = KNN
            KNN_params = [wei, nei]
            
            
# AD
print("Treinando AD...")
AD_melhor = DecisionTreeClassifier()
AD_maior = -1
AD_params = []

for cri in ['gini', 'entropy', 'log_loss']:
    for spl in ['best', 'random']:
        for mde in [1, 3, 5, 7, 9]:
            for mss in [2, 4, 6, 8]:
                for msl in [1, 2, 3]:
                    print("\tTreinando ", [cri, spl, mde, mss, msl])
                    AD = DecisionTreeClassifier(criterion = cri, splitter = spl, max_depth = mde, min_samples_split = mss, min_samples_leaf = msl)
                    AD.fit(x_treino, y_treino)
                    opiniao = AD.predict(x_validacao)
                    Acc = accuracy_score(y_validacao, opiniao)
                    if (Acc > AD_maior):
                        AD_maior = Acc
                        AD_melhor = AD
                        AD_params = [cri, spl, mde, mss, msl]

# NB
print("Treinando NB...")
NB_melhor = GaussianNB()
NB_maior = -1

NB_melhor = GaussianNB()
NB_melhor.fit(x_treino, y_treino)
NB_maior = accuracy_score(y_validacao, NB_melhor.predict(x_validacao))

# SVM
print("Treinando SVM...")
SVM_melhor = SVC()
SVM_maior = -1
SVM_params = []

for Cval in [0.5, 1.0, 1.5, 2.0]:
    for ker in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        print("\tTreinando ", [Cval, ker])
        SVM = SVC()
        SVM.fit(x_treino, y_treino)
        opiniao = SVM.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        if (Acc > SVM_maior):
            SVM_maior = Acc
            SVM_melhor = SVM
            SVM_params = [Cval, ker]

# MLP
print("Treinando MLP...")
MLP_melhor = MLPClassifier()
MLP_maior = -1
MLP_params = []

for mit in [100, 200]:
    for lra in ['constant', 'invscaling', 'adaptive']:
        for hla in [1, 2]:
            for npl in [13, 26]:
                for act in ['identity', 'logistic', 'tanh', 'relu']:
                    for bsi in [32, 48]:
                        print("\tTreinando ", [mit, lra, hla, npl, act, bsi])
                        hls = ()
                        for i in range(hla):
                            hls += (npl,)
                        MLP = MLPClassifier(max_iter = mit, learning_rate = lra, hidden_layer_sizes = hls, activation = act, batch_size = bsi)
                        MLP.fit(x_treino, y_treino)
                        opiniao = MLP.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)
                        if (Acc > MLP_maior):
                            MLP_maior = Acc
                            MLP_melhor = MLP
                            MLP_params = [mit, lra, hla, npl, act, bsi]

# Random Forest
print("Treinando RF...")
RF_melhor = RandomForestClassifier()
RF_maior = -1
RF_params = []

for nes in [50, 100, 150, 200]:
    for cri in ['gini', 'entropy', 'log_loss']:
        for mde in [1, 3, 5, 7, 9]:
            for mss in [2, 4, 6, 8]:
                for msl in [1, 2, 3]:
                    print("\tTreinando ", [nes, cri, mde, mss, msl])
                    RF = RandomForestClassifier(n_estimators = nes, criterion = cri, max_depth = mde, min_samples_split = mss, min_samples_leaf = msl)
                    RF.fit(x_treino, y_treino)
                    opiniao = RF.predict(x_validacao)
                    Acc = accuracy_score(y_validacao, opiniao)
                    if (Acc > RF_maior):
                        RF_maior = Acc
                        RF_melhor = RF
                        RF_params = [nes, cri, mde, mss, msl]

# Bagging
print("Treinando Bagging...")
BG_melhor = BaggingClassifier
BG_maior = -1
BG_params = []

for nes in [5, 10, 15, 20]:
    for msa in [0.25, 0.5, 0.75, 1.0]:
        for est in [KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), MLPClassifier()]:
            print("\tTreinando ", [nes, msa, est])
            BG = BaggingClassifier(n_estimators = nes, max_samples = msa, estimator = est)
            BG.fit(x_treino, y_treino)
            opiniao = BG.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            if (Acc > BG_maior):
                BG_maior = Acc
                BG_melhor = BG
                BG_params = [nes, msa, est]


# Boosting
print("Treinando Boosting...")
BO_melhor = AdaBoostClassifier()
BO_maior = -1
BO_params = []

for nes in [5, 10, 15, 20]:
    for est in [DecisionTreeClassifier(), SVC(), MLPClassifier()]:
        for lra in [0.1, 0.5, 1.0, 1.5, 2.0]:
            print("\tTreinando ", [nes, est, lra])
            BO = AdaBoostClassifier(n_estimators = nes, estimator = est, learning_rate = lra)
            BO.fit(x_treino, y_treino)
            opiniao = BO.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            if (Acc > BO_maior):
                BO_maior = Acc
                BO_melhor = BO
                BO_params = [nes, est, lra]

# ------------------------------
# PRINT DOS RESULTADOS
# ------------------------------


# ==============================
# SESSÃO FINAL DE APRESENTAÇÃO
# ==============================

print("\n" + "="*70)
print("RESULTADOS DA OTIMIZAÇÃO DE HIPERPARÂMETROS (Validação)")
print("="*70)

# Dicionário para consolidar os resultados
resultados_valid = {}

# KNN
if KNN_maior != -1:
    resultados_valid['KNN'] = {
        'acc': KNN_maior,
        'params': f'weights={KNN_params[0]}, n_neighbors={KNN_params[1]}'
    }

# Decision Tree (AD)
if AD_maior != -1:
    resultados_valid['Decision Tree'] = {
        'acc': AD_maior,
        'params': f'criterion={AD_params[0]}, splitter={AD_params[1]}, max_depth={AD_params[2]}, min_samples_split={AD_params[3]}, min_samples_leaf={AD_params[4]}'
    }

# Naive Bayes (sem busca)
if NB_maior != -1:
    resultados_valid['Naive Bayes'] = {
        'acc': NB_maior,
        'params': 'default'
    }

# SVM
if SVM_maior != -1:
    resultados_valid['SVM'] = {
        'acc': SVM_maior,
        'params': f'C={SVM_params[0]}, kernel={SVM_params[1]}'
    }

# MLP
if MLP_maior != -1:
    resultados_valid['MLP'] = {
        'acc': MLP_maior,
        'params': f'max_iter={MLP_params[0]}, learning_rate={MLP_params[1]}, hidden_layers={MLP_params[2]} camadas de {MLP_params[3]}, activation={MLP_params[4]}, batch_size={MLP_params[5]}'
    }

# Random Forest
if RF_maior != -1:
    resultados_valid['Random Forest'] = {
        'acc': RF_maior,
        'params': f'n_estimators={RF_params[0]}, criterion={RF_params[1]}, max_depth={RF_params[2]}, min_samples_split={RF_params[3]}, min_samples_leaf={RF_params[4]}'
    }

# Bagging
if BG_maior != -1:
    resultados_valid['Bagging'] = {
        'acc': BG_maior,
        'params': f'n_estimators={BG_params[0]}, max_samples={BG_params[1]:.0f}, base_estimator={type(BG_params[2]).__name__}'
    }

# AdaBoost
if BO_maior != -1:
    resultados_valid['AdaBoost'] = {
        'acc': BO_maior,
        'params': f'n_estimators={BO_params[0]}, base_estimator={type(BO_params[1]).__name__}, learning_rate={BO_params[2]}'
    }

# Impressão da tabela de validação
print(f"{'Modelo':20} | {'Acurácia (Val)':15} | {'Melhores Hiperparâmetros'}")
print("-"*70)
for nome, info in resultados_valid.items():
    print(f"{nome:20} | {info['acc']:15.4f} | {info['params']}")

# ------------------------------------------
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ------------------------------------------
print("\n" + "="*70)
print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (modelos ajustados com treino+validação)")
print("="*70)

# Combinando treino e validação para treinar o modelo final
x_train_final = np.vstack([x_treino, x_validacao])
y_train_final = np.hstack([y_treino, y_validacao])

# Dicionário com os melhores modelos (já treinados apenas na validação, vamos re-treinar)
melhores_modelos = {
    'KNN': KNN_melhor,
    'Decision Tree': AD_melhor,
    'Naive Bayes': NB_melhor,
    'SVM': SVM_melhor,
    'MLP': MLP_melhor,
    'Random Forest': RF_melhor,
    'Bagging': BG_melhor,
    'AdaBoost': BO_melhor
}

# Treinamento final e avaliação
resultados_teste = {}
for nome, modelo in melhores_modelos.items():
    modelo.fit(x_train_final, y_train_final)
    y_pred = modelo.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    resultados_teste[nome] = {'acc': acc, 'f1': f1}

# Exibição final
print(f"{'Modelo':20} | {'Acurácia (Teste)':15} | {'F1-score (Teste)':15}")
print("-"*55)
for nome, metrics in resultados_teste.items():
    print(f"{nome:20} | {metrics['acc']:15.4f} | {metrics['f1']:15.4f}")

# (Opcional) Matriz de confusão do melhor modelo
melhor_modelo_nome = max(resultados_teste, key=lambda x: resultados_teste[x]['acc'])
melhor_modelo = melhores_modelos[melhor_modelo_nome]
y_pred_best = melhor_modelo.predict(x_teste)

plt.figure(figsize=(6,5))
cm = confusion_matrix(y_teste, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=melhor_modelo.classes_, yticklabels=melhor_modelo.classes_)
plt.title(f'Matriz de Confusão - {melhor_modelo_nome} (Teste)')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

print(f"\nMelhor modelo no teste: {melhor_modelo_nome} com acurácia {resultados_teste[melhor_modelo_nome]['acc']:.4f}")