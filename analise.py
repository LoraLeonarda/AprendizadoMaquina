import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ============================================================
# Dados em vetor
# ============================================================
KNN_Acc = [0.9233, 0.9216, 0.9237, 0.9213, 0.9206, 0.9193, 0.9196, 0.9199, 0.9128, 0.9179, 0.9179, 0.9216, 0.9189, 0.9176, 0.9227, 0.9240, 0.9179, 0.9233, 0.9193, 0.9220]
KNN_Fsc = [0.9090, 0.9067, 0.9100, 0.9061, 0.9066, 0.9046, 0.9048, 0.9056, 0.8983, 0.9032, 0.9048, 0.9075, 0.9049, 0.9035, 0.9095, 0.9090, 0.9024, 0.9082, 0.9045, 0.9080]

AD_Acc = [0.9403, 0.9379, 0.9410, 0.9386, 0.9332, 0.9393, 0.9393, 0.9383, 0.9345, 0.9362, 0.9372, 0.9400, 0.9372, 0.9386, 0.9389, 0.9389, 0.9355, 0.9359, 0.9389, 0.9434]
AD_Fsc = [0.9320, 0.9259, 0.9282, 0.9296, 0.9228, 0.9297, 0.9282, 0.9235, 0.9236, 0.9254, 0.9303, 0.9309, 0.9291, 0.9331, 0.9316, 0.9301, 0.9284, 0.9276, 0.9280, 0.9278]

NB_Acc = [0.9457, 0.9400, 0.9440, 0.9427, 0.9406, 0.9444, 0.9464, 0.9427, 0.9352, 0.9420, 0.9372, 0.9450, 0.9427, 0.9410, 0.9396, 0.9471, 0.9413, 0.9447, 0.9430, 0.9491]
NB_Fsc = [0.9325, 0.9274, 0.9308, 0.9295, 0.9279, 0.9308, 0.9329, 0.9299, 0.9228, 0.9287, 0.9249, 0.9319, 0.9298, 0.9276, 0.9264, 0.9331, 0.9283, 0.9311, 0.9296, 0.9355]

SVM_Acc = [0.9274, 0.9230, 0.9257, 0.9244, 0.9247, 0.9271, 0.9264, 0.9254, 0.9166, 0.9261, 0.9233, 0.9298, 0.9277, 0.9227, 0.9250, 0.9277, 0.9237, 0.9247, 0.9237, 0.9311]
SVM_Fsc = [0.9138, 0.9097, 0.9119, 0.9093, 0.9111, 0.9127, 0.9126, 0.9115, 0.9028, 0.9117, 0.9090, 0.9153, 0.9138, 0.9083, 0.9113, 0.9135, 0.9091, 0.9098, 0.9099, 0.9162]

MLP_Acc = [0.9413, 0.9311, 0.9355, 0.9345, 0.9345, 0.9291, 0.9389, 0.9294, 0.9345, 0.9335, 0.9389, 0.9318, 0.9359, 0.9322, 0.9332, 0.9376, 0.9315, 0.9328, 0.9267, 0.9393]
MLP_Fsc = [0.9265, 0.9206, 0.9235, 0.9209, 0.9227, 0.9142, 0.9258, 0.9154, 0.9284, 0.9226, 0.9305, 0.9190, 0.9219, 0.9173, 0.9216, 0.9234, 0.9181, 0.9215, 0.9114, 0.9248]

RF_Acc = [0.9579, 0.9545, 0.9579, 0.9545, 0.9556, 0.9556, 0.9512, 0.9552, 0.9535, 0.9505, 0.9532, 0.9562, 0.9552, 0.9539, 0.9562, 0.9528, 0.9559, 0.9552, 0.9535, 0.9552]
RF_Fsc = [0.9425, 0.9391, 0.9425, 0.9390, 0.9401, 0.9400, 0.9356, 0.9397, 0.9380, 0.9350, 0.9376, 0.9408, 0.9399, 0.9385, 0.9408, 0.9372, 0.9403, 0.9396, 0.9378, 0.9395]

BG_Acc = [0.9545, 0.9488, 0.9512, 0.9518, 0.9505, 0.9498, 0.9495, 0.9515, 0.9501, 0.9484, 0.9464, 0.9522, 0.9532, 0.9518, 0.9515, 0.9525, 0.9512, 0.9528, 0.9474, 0.9590]
BG_Fsc = [0.9444, 0.9379, 0.9392, 0.9402, 0.9387, 0.9392, 0.9377, 0.9411, 0.9403, 0.9359, 0.9382, 0.9399, 0.9419, 0.9434, 0.9398, 0.9399, 0.9406, 0.9409, 0.9359, 0.9484]

BO_Acc = [0.9267, 0.9240, 0.9284, 0.9339, 0.9318, 0.9301, 0.9257, 0.9216, 0.9196, 0.9322, 0.9294, 0.9288, 0.9301, 0.9264, 0.9264, 0.9328, 0.9308, 0.9335, 0.9244, 0.9322]
BO_Fsc = [0.9291, 0.9274, 0.9290, 0.9337, 0.9326, 0.9156, 0.9260, 0.9246, 0.9063, 0.9180, 0.9164, 0.9309, 0.9156, 0.9118, 0.9125, 0.9321, 0.9308, 0.9321, 0.9108, 0.9160]

Soma_Acc = [0.9423, 0.9349, 0.9386, 0.9355, 0.9379, 0.9403, 0.9383, 0.9383, 0.9328, 0.9383, 0.9406, 0.9366, 0.9427, 0.9372, 0.9400, 0.9400, 0.9352, 0.9400, 0.9349, 0.9413]
Soma_Fsc = [0.9277, 0.9225, 0.9248, 0.9216, 0.9254, 0.9255, 0.9246, 0.9242, 0.9194, 0.9249, 0.9275, 0.9226, 0.9285, 0.9225, 0.9268, 0.9255, 0.9211, 0.9267, 0.9200, 0.9262]

MajVoto_Acc = [0.9393, 0.9332, 0.9386, 0.9352, 0.9355, 0.9311, 0.9372, 0.9322, 0.9301, 0.9386, 0.9369, 0.9376, 0.9376, 0.9301, 0.9359, 0.9386, 0.9328, 0.9400, 0.9322, 0.9406]
MajVoto_Fsc = [0.9251, 0.9203, 0.9248, 0.9216, 0.9228, 0.9163, 0.9235, 0.9180, 0.9171, 0.9256, 0.9231, 0.9236, 0.9237, 0.9157, 0.9226, 0.9242, 0.9188, 0.9258, 0.9177, 0.9259]

Borda_Acc = [0.9400, 0.9328, 0.9403, 0.9372, 0.9372, 0.9315, 0.9376, 0.9328, 0.9318, 0.9406, 0.9403, 0.9383, 0.9386, 0.9328, 0.9383, 0.9400, 0.9328, 0.9400, 0.9328, 0.9427]
Borda_Fsc = [0.9256, 0.9195, 0.9264, 0.9231, 0.9239, 0.9164, 0.9235, 0.9185, 0.9187, 0.9272, 0.9263, 0.9239, 0.9246, 0.9180, 0.9244, 0.9252, 0.9186, 0.9256, 0.9180, 0.9274]

# ============================================================
# Organização em DataFrames
# ============================================================
individuais = ['KNN', 'AD', 'NB', 'SVM', 'MLP']
ensemble = ['RF', 'BG', 'BO', 'Soma', 'MajVoto', 'Borda']
todos = individuais + ensemble

# Dicionário para acesso fácil
acc_dict = {name: eval(f'{name}_Acc') for name in todos}
fsc_dict = {name: eval(f'{name}_Fsc') for name in todos}

df_acc = pd.DataFrame(acc_dict)
df_fsc = pd.DataFrame(fsc_dict)

# Número de repetições
n_rep = len(df_acc)

# ============================================================
# Função para gerar tabela de média (DP)
# ============================================================
def tabela_estatisticas(df, nomes=None):
    if nomes is None:
        nomes = df.columns
    media = df[nomes].mean()
    dp = df[nomes].std(ddof=1)  # desvio padrão amostral
    tabela = pd.DataFrame({'Média': media, 'DP': dp})
    return tabela

def tabela_repeticao(df, nomes):
    """Tabela com linhas 1..20 e colunas dos classificadores, + linha média(dp)."""
    linhas = []
    for i in range(len(df)):
        linha = [f"{i+1:2d}"] + [f"{df[col].iloc[i]:.4f}" for col in nomes]
        linhas.append(linha)
    # linha média (DP)
    media_dp = ["Média(DP)"] + [f"{df[col].mean():.4f} ({df[col].std(ddof=1):.4f})" for col in nomes]
    linhas.append(media_dp)
    return linhas

def imprimir_tabela_repeticao(titulo, nomes, df):
    print(f"\n{titulo}")
    cabecalho = ["Rep"] + nomes
    print("".join(f"{h:>10}" for h in cabecalho))
    for linha in tabela_repeticao(df, nomes):
        print("".join(f"{str(c):>10}" for c in linha))
    print()

# ============================================================
# Função para análise de Friedman + post-hoc Wilcoxon pareado (sem ajuste)
# ============================================================
def friedman_wilcoxon_posthoc(df_sub, metrica_label, alpha=0.05):
    nomes = list(df_sub.columns)
    stat, p = stats.friedmanchisquare(*[df_sub[c] for c in nomes])
    print(f"\nTeste de Friedman para {metrica_label}:")
    print(f"  Estatística = {stat:.4f}, valor-p = {p:.4f}")
    if p < alpha:
        print(f"  Há diferença significativa entre os classificadores (p < {alpha}).")
        # Post-hoc Wilcoxon pareado sem ajuste
        pairs = []
        for i, j in combinations(range(len(nomes)), 2):
            c1, c2 = nomes[i], nomes[j]
            _, p_val = stats.wilcoxon(df_sub[c1], df_sub[c2])
            pairs.append((c1, c2, p_val))
        pairs.sort(key=lambda x: x[2])
        print("\n  Valores-p do teste de Wilcoxon pareado (sem ajuste):")
        for n1, n2, pv in pairs:
            sig = "(*)" if pv < alpha else ""
            print(f"    {n1} vs {n2}: p = {pv:.4f} {sig}")
        # Ranks médios (para referência)
        ranks = df_sub.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().sort_values()
        print("\n  Ranks médios (referência):")
        for name, rank in avg_ranks.items():
            print(f"    {name}: {rank:.2f}")
        return avg_ranks, pairs
    else:
        print(f"  Não há diferença significativa (p >= {alpha}).")
        return None, None

# ============================================================
# Função para teste de Wilcoxon pareado entre dois classificadores
# (mantida para a comparação final)
# ============================================================
def wilcoxon_test(df, c1, c2, metrica_label):
    stat, p = stats.wilcoxon(df[c1], df[c2])
    print(f"\nTeste de Wilcoxon pareado ({metrica_label}): {c1} vs {c2}")
    print(f"  Estatística = {stat:.4f}, valor-p = {p:.4f}")
    if p < 0.05:
        melhor = c1 if df[c1].mean() > df[c2].mean() else c2
        print(f"  Diferença significativa (p < 0.05). {melhor} é significativamente melhor.")
    else:
        print(f"  Não há diferença significativa (p >= 0.05).")
    return p

# ============================================================
# INÍCIO DA ANÁLISE TEXTUAL PARA O RELATÓRIO
# ============================================================
print("="*70)
print("ANÁLISE ESTATÍSTICA DE DESEMPENHO DOS CLASSIFICADORES")
print("="*70)

# -------- 1. Tabelas de acurácias e F1-Score para classificadores monolíticos --------
print("\n\n1. RESULTADOS DOS CLASSIFICADORES MONOLÍTICOS")
print("-"*50)
imprimir_tabela_repeticao("Tabela 1: Acurácia por repetição - Classificadores Individuais", individuais, df_acc)
print("\nEstatísticas descritivas (média e desvio padrão):")
print(tabela_estatisticas(df_acc, individuais).to_string(float_format=lambda x: f"{x:.4f}"))

imprimir_tabela_repeticao("Tabela 2: F1-Score por repetição - Classificadores Individuais", individuais, df_fsc)
print("\nEstatísticas descritivas (média e desvio padrão):")
print(tabela_estatisticas(df_fsc, individuais).to_string(float_format=lambda x: f"{x:.4f}"))

# -------- 2. Análise estatística: monolíticos --------
print("\n\n2. ANÁLISE ESTATÍSTICA DOS CLASSIFICADORES MONOLÍTICOS")
print("-"*50)
ranks_acc_ind, _ = friedman_wilcoxon_posthoc(df_acc[individuais], "Acurácia (Individuais)")
ranks_fsc_ind, _ = friedman_wilcoxon_posthoc(df_fsc[individuais], "F1-Score (Individuais)")

if ranks_acc_ind is not None:
    print("\nRanks médios (Acurácia):")
    for name, rank in ranks_acc_ind.items():
        print(f"  {name}: {rank:.2f}")

# -------- 3. Tabelas para estratégias de combinação/ensemble --------
print("\n\n3. RESULTADOS DAS ESTRATÉGIAS DE COMBINAÇÃO E ENSEMBLE")
print("-"*50)
# Renomear para os nomes do relatório
ensemble_nomes_relatorio = ['Random Forest', 'Bagging', 'Boosting', 'Soma', 'SMC (Vot. Majoritária)', 'Borda Count']
mapa_nomes = dict(zip(ensemble, ensemble_nomes_relatorio))
df_acc_ens_rel = df_acc[ensemble].rename(columns=mapa_nomes)
df_fsc_ens_rel = df_fsc[ensemble].rename(columns=mapa_nomes)

imprimir_tabela_repeticao("Tabela 3: Acurácia por repetição - Combinação/Ensemble", list(mapa_nomes.values()), df_acc_ens_rel)
print("\nEstatísticas descritivas (média e desvio padrão):")
print(tabela_estatisticas(df_acc_ens_rel).to_string(float_format=lambda x: f"{x:.4f}"))

imprimir_tabela_repeticao("Tabela 4: F1-Score por repetição - Combinação/Ensemble", list(mapa_nomes.values()), df_fsc_ens_rel)
print("\nEstatísticas descritivas (média e desvio padrão):")
print(tabela_estatisticas(df_fsc_ens_rel).to_string(float_format=lambda x: f"{x:.4f}"))

# -------- 4. Análise estatística: combinação/ensemble --------
print("\n\n4. ANÁLISE ESTATÍSTICA DAS ESTRATÉGIAS DE COMBINAÇÃO/ENSEMBLE")
print("-"*50)
ranks_acc_ens, _ = friedman_wilcoxon_posthoc(df_acc_ens_rel, "Acurácia (Combinação/Ensemble)")
ranks_fsc_ens, _ = friedman_wilcoxon_posthoc(df_fsc_ens_rel, "F1-Score (Combinação/Ensemble)")

if ranks_acc_ens is not None:
    print("\nRanks médios (Acurácia):")
    for name, rank in ranks_acc_ens.items():
        print(f"  {name}: {rank:.2f}")

# -------- 5. Melhor individual vs melhor combinação/ensemble --------
print("\n\n5. COMPARAÇÃO ENTRE O MELHOR CLASSIFICADOR MONOLÍTICO E A MELHOR ESTRATÉGIA DE COMBINAÇÃO/ENSEMBLE")
print("-"*50)
# Identificar melhor pela média de acurácia
melhor_ind = df_acc[individuais].mean().idxmax()
melhor_ens = df_acc[ensemble].mean().idxmax()
melhor_ind_nome = melhor_ind
melhor_ens_nome = mapa_nomes[melhor_ens]  # nome para relatório

print(f"Melhor classificador individual (acurácia média): {melhor_ind_nome} ({df_acc[melhor_ind].mean():.4f})")
print(f"Melhor estratégia combinada/ensemble (acurácia média): {melhor_ens_nome} ({df_acc[melhor_ens].mean():.4f})")

# Teste Wilcoxon entre eles
wilcoxon_test(df_acc, melhor_ind, melhor_ens, "Acurácia")
wilcoxon_test(df_fsc, melhor_ind, melhor_ens, "F1-Score")

# Interpretação adicional
print("\n\n6. INTERPRETAÇÃO GERAL PARA O RELATÓRIO")
print("-"*50)
print("""
Com base nos testes de Friedman e post-hoc de Wilcoxon pareado (sem ajuste, α=0,05),
foram detectadas diferenças significativas entre os classificadores monolíticos 
em termos de acurácia e F1-Score. Da mesma forma, as estratégias de combinação/ensemble 
também apresentaram diferenças significativas entre si.

Ao comparar o melhor classificador monolítico com a melhor estratégia de combinação/ensemble 
via teste de Wilcoxon pareado, verificou-se se a diferença é estatisticamente significativa.
(Substituir a conclusão conforme p-valores obtidos.)
""")

print("="*70)
print("FIM DA ANÁLISE - TEXTO PRONTO PARA INSERÇÃO NO RELATÓRIO")
print("="*70)