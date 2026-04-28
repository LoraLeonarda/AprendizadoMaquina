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
# PASTA PARA MATRIZES DE CONFUSÃO
# ------------------------------
output_dir = "matrizes_confusao"
os.makedirs(output_dir, exist_ok=True)

def salvar_matriz_confusao(nome_modelo, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Store', 'Online', 'Hybrid'],
                yticklabels=['Store', 'Online', 'Hybrid'])
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.tight_layout()
    nome_arquivo = f"{nome_modelo.replace(' ', '_')}_matriz_confusao.png"
    caminho_completo = os.path.join(output_dir, nome_arquivo)
    plt.savefig(caminho_completo, dpi=150)
    plt.close()
    #print(f"Matriz de confusão salva: {caminho_completo}")

# ------------------------------
# AVALIAÇÃO COM PARÂMETROS DEFAULT
# ------------------------------
modelos_default = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
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
    salvar_matriz_confusao(nome, y_teste, opiniao)


# ------------------------------
# BUSCA DE HIPERPARÂMETROS
# ------------------------------
print("\n=== Busca de Hiperparâmetros com GridSearchCV ===\n")

# Definir as grades para cada modelo
param_grids = {
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    "SVM": {
        # tava quebrando
    },
    "Naive Bayes": {
        'var_smoothing': np.logspace(-9, -5, 5)  # GaussianNB tem poucos parâmetros
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    "Bagging": {
        # talvez?
    },
    "AdaBoost": {
        # talvez?
    }
}

# Dicionário para armazenar os melhores modelos e resultados
melhores_modelos = {}

# Para cada tipo de modelo, realiza o GridSearchCV
for nome, grid in param_grids.items():
    print(f"\nTreinando {nome}...")
    
    # Instanciar o modelo base com random_state fixo (quando aplicável)
    if nome == "KNN":
        modelo_base = KNeighborsClassifier()
    elif nome == "Decision Tree":
        modelo_base = DecisionTreeClassifier(random_state=42)
    elif nome == "MLP":
        modelo_base = MLPClassifier(max_iter=1000, random_state=42)
    elif nome == "SVM":
        modelo_base = SVC(random_state=42)
    elif nome == "Naive Bayes":
        modelo_base = GaussianNB()
    elif nome == "Random Forest":
        modelo_base = RandomForestClassifier(random_state=42)
    elif nome == "Bagging":
        # BaggingClassifier aceita um estimator base; aqui usamos árvore
        modelo_base = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    elif nome == "AdaBoost":
        modelo_base = AdaBoostClassifier(random_state=42)
    else:
        continue

    # Configurar o GridSearchCV
    gs = GridSearchCV(
        modelo_base,
        param_grid=grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Executar a busca no conjunto de treino
    gs.fit(x_treino, y_treino)
    
    # Melhor modelo encontrado
    melhor_modelo = gs.best_estimator_
    melhores_modelos[nome] = melhor_modelo
    
    # Resultados
    print(f"Melhores parâmetros para {nome}: {gs.best_params_}")
    print(f"Melhor acurácia média (validação cruzada): {gs.best_score_:.4f}")
    
    # Avaliar no conjunto de validação (opcional)
    pred_val = melhor_modelo.predict(x_validacao)
    acc_val = accuracy_score(y_validacao, pred_val)
    f1_val = f1_score(y_validacao, pred_val, average='weighted')
    print(f"Desempenho na validação: Acurácia = {acc_val:.4f} | F1 = {f1_val:.4f}")

# ------------------------------
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ------------------------------
print("\n=== Resultados finais no conjunto de teste (melhores hiperparâmetros) ===\n")
for nome, modelo in melhores_modelos.items():
    pred_teste = modelo.predict(x_teste)
    acc_teste = accuracy_score(y_teste, pred_teste)
    f1_teste = f1_score(y_teste, pred_teste, average='weighted')
    print(f"{nome:15} | Acurácia: {acc_teste:.4f} | F1-score: {f1_teste:.4f}")
    # Salvar matriz de confusão do melhor modelo
    salvar_matriz_confusao(f"{nome}_tuned", y_teste, pred_teste)