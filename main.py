# Bibliotecas basicas
import pandas as pd
import numpy as np

# Métodos de split e métricas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# Junção de modelos pela soma
def sum_rule(probas_list):
    prob_sum = np.sum(probas_list, axis=0)
    return np.argmax(prob_sum, axis=1)

# Junção de modelos por voto majoritário (moda)
def majority_voting(classifiers, X):
    preds = np.array([clf.predict(X) for clf in classifiers])
    majority, _ = mode(preds, axis=0)
    return majority.ravel()

# Borda Count (cortesia DeepSeek)
def borda_count(probas_list):
    n_amostras = probas_list[0].shape[0]
    n_classes = probas_list[0].shape[1]
    borda_scores = np.zeros((n_amostras, n_classes))
    for probas in probas_list:
        # Ordena os índices das classes por probabilidade decrescente para cada amostra
        sorted_idx = np.argsort(-probas, axis=1)  # (n_amostras, n_classes)
        # Atribui pontuação: posição 0 -> n_classes, posição 1 -> n_classes-1, ...
        points = np.arange(n_classes, 0, -1)  # [n_classes, n_classes-1, ..., 1]
        for i in range(n_amostras):
            borda_scores[i, sorted_idx[i]] += points
    return np.argmax(borda_scores, axis=1)
    

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

# 20 execuções para não variar o resultado final!
for repeticao in range(20):

    # Divisão estratificada: 50% treino, 25% validação, 25% teste
    x_treino, x_temp, y_treino, y_temp = train_test_split(
        X, Y, test_size=0.5, stratify=Y, random_state=42
    )
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # ------------------------------
    # BUSCA DE HIPERPARÂMETROS
    # ------------------------------

    # KNN
    print("Treinando KNN...")
    KNN_melhor = KNeighborsClassifier()
    KNN_maior = -1
    KNN_params = []

    for wei in ['uniform', 'distance']:
        for nei in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            #print("\tTreinando ", [wei, nei])
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
                        #print("\tTreinando ", [cri, spl, mde, mss, msl])
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
        for ker in ['poly', 'rbf', 'sigmoid']:
            #print("\tTreinando ", [Cval, ker])
            SVM = SVC(C = Cval, kernel = ker, probability = True)
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
                            #print("\tTreinando ", [mit, lra, hla, npl, act, bsi])
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
                        #print("\tTreinando ", [nes, cri, mde, mss, msl])
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
            for est in [KNeighborsClassifier(), DecisionTreeClassifier(), MLPClassifier()]:
                #print("\tTreinando ", [nes, msa, est])
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
        for est in [DecisionTreeClassifier(), MLPClassifier()]:
            for lra in [0.1, 0.5, 1.0, 1.5, 2.0]:
                #print("\tTreinando ", [nes, est, lra])
                BO = AdaBoostClassifier(n_estimators = nes, estimator = est, learning_rate = lra)
                BO.fit(x_treino, y_treino)
                opiniao = BO.predict(x_validacao)
                Acc = accuracy_score(y_validacao, opiniao)
                if (Acc > BO_maior):
                    BO_maior = Acc
                    BO_melhor = BO
                    BO_params = [nes, est, lra]

    # Obter probabilidades para cada classificador
    classificadores = [KNN_melhor, AD_melhor, NB_melhor, SVM_melhor, MLP_melhor]
    probas = [clf.predict_proba(x_teste) for clf in classificadores]

    # Predições combinadas
    y_pred_sum = sum_rule(probas)
    y_pred_maj = majority_voting(classificadores, x_teste)
    y_pred_borda = borda_count(probas)

    # Métricas para cada combinação
    acc_sum = accuracy_score(y_teste, y_pred_sum)
    f1_sum = f1_score(y_teste, y_pred_sum, average='weighted')

    acc_maj = accuracy_score(y_teste, y_pred_maj)
    f1_maj = f1_score(y_teste, y_pred_maj, average='weighted')

    acc_borda = accuracy_score(y_teste, y_pred_borda)
    f1_borda = f1_score(y_teste, y_pred_borda, average='weighted')


    # ------------------------------
    # PRINT DOS RESULTADOS
    # ------------------------------
    print(f"Repeticao {repeticao+1}:")

    # KNN
    y_pred = KNN_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    KNN: Acc: {acc:.4f} F1: {f1:.4f} | Params: weights={KNN_params[0]}, n_neighbors={KNN_params[1]}")

    # AD (Decision Tree)
    y_pred = AD_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    AD: Acc: {acc:.4f} F1: {f1:.4f} | Params: criterion={AD_params[0]}, splitter={AD_params[1]}, max_depth={AD_params[2]}, min_samples_split={AD_params[3]}, min_samples_leaf={AD_params[4]}")

    # NB
    y_pred = NB_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    NB: Acc: {acc:.4f} F1: {f1:.4f} (hiperparâmetros padrão)")

    # SVM
    y_pred = SVM_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    SVM: Acc: {acc:.4f} F1: {f1:.4f} | Params: C={SVM_params[0]}, kernel={SVM_params[1]}")

    # MLP
    y_pred = MLP_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    MLP: Acc: {acc:.4f} F1: {f1:.4f} | Params: max_iter={MLP_params[0]}, learning_rate={MLP_params[1]}, hidden_layers={MLP_params[2]}x{MLP_params[3]} neurons, activation={MLP_params[4]}, batch_size={MLP_params[5]}")

    # Random Forest
    y_pred = RF_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    RF: Acc: {acc:.4f} F1: {f1:.4f} | Params: n_estimators={RF_params[0]}, criterion={RF_params[1]}, max_depth={RF_params[2]}, min_samples_split={RF_params[3]}, min_samples_leaf={RF_params[4]}")

    # Bagging
    y_pred = BG_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    BG: Acc: {acc:.4f} F1: {f1:.4f} | Params: n_estimators={BG_params[0]}, max_samples={BG_params[1]}, estimator={type(BG_params[2]).__name__}")

    # Boosting (AdaBoost)
    y_pred = BO_melhor.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, average='weighted')
    print(f"    BO: Acc: {acc:.4f} F1: {f1:.4f} | Params: n_estimators={BO_params[0]}, estimator={type(BO_params[1]).__name__}, learning_rate={BO_params[2]}")
    
    # Combinações
    print(f"    Soma: Acc: {acc_sum:.4f} F1: {f1_sum:.4f}")
    print(f"    MajVoto: Acc: {acc_maj:.4f} F1: {f1_maj:.4f}")
    print(f"    Borda: Acc: {acc_borda:.4f} F1: {f1_borda:.4f}")

    print()  # linha em branco entre repetições
    
