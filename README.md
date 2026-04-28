# Classificação de Preferência de Compras

Este projeto realiza a classificação da preferência de compra (`shopping_preference`) — **Store**, **Online** ou **Hybrid** — utilizando diversos algoritmos de aprendizado de máquina.

## Visão Geral

O código segue estas etapas:

1. **Carregamento e pré‑processamento**  
   Lê `ShoppingDataset.csv`, mapeia variáveis categóricas (`gender`, `city_tier`, `shopping_preference`) para números e separa features (X) do target (Y).

2. **Divisão estratificada**  
   Divide os dados em **treino (50%)**, **validação (25%)** e **teste (25%)** mantendo a proporção das classes.

3. **Avaliação com hiperparâmetros padrão**  
   Treina oito classificadores (KNN, Árvore de Decisão, MLP, SVM, Naïve Bayes, Random Forest, Bagging, AdaBoost) com configurações default.  
   Calcula acurácia e F1‑score no teste e salva as matrizes de confusão na pasta `matrizes_confusao/`.

4. **Busca de hiperparâmetros (GridSearchCV)**  
   Aplica validação cruzada (5‑fold) para ajustar os modelos que possuem grade definida (KNN, Decision Tree, MLP, Naïve Bayes, Random Forest).  
   Exibe os melhores parâmetros e o desempenho na validação.

5. **Avaliação final**  
   Utiliza os melhores estimadores encontrados para prever o conjunto de teste, reportando acurácia e F1‑score, e gera as respectivas matrizes de confusão (sufixo `_tuned`).

## Dependências

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit‑learn

## Como Executar

1. Certifique‑se de que o arquivo `ShoppingDataset.csv` está no mesmo diretório.
2. Execute o script principal:
   ```bash
   python nome_do_script.py