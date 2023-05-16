# %% [markdown]
# # <center> <img src="figs/LogoUFSCar.jpg" alt="Logo UFScar" width="110" align="left"/>  <br/> <center>Universidade Federal de São Carlos (UFSCar)<br/><font size="4"> Departamento de Computação, campus Sorocaba</center></font>
# </p>
# 
# <br/>
# <font size="4"><center><b>Disciplina: Aprendizado de Máquina</b></center></font>
#   
# <font size="3"><center>Prof. Dr. Tiago A. Almeida</center></font>
# 
# <br/>
# <br/>
# 
# <center><i><b>
# Atenção: não são autorizadas cópias, divulgações ou qualquer tipo de uso deste material sem o consentimento prévio dos autores.
# </center></i></b>

# %% [markdown]
# # <center>Exercício - Sistemas de Recomendação</center>
# 
# Introdução
# ----------
# Neste *notebook*, você fará um protocolo experimental completo no contexto de Sistemas de Recomendação, aplicando dois algoritmos de Filtragem Colaborativa, KNN e SVD, sobre uma base real de recomendação de filmes. Os experimentos que serão apresentados foram projetados com o intuito de facilitar o entendimento da área e serem genéricos, permitindo sua reprodução em outros domínios de aplicação.
# 
# Antes de começar, é recomendável que você revise os conceitos apresentados em aula.
# 
# 
# ## Instruções
# Este arquivo contém o código que auxiliará no desenvolvimento do exercício. Você precisará completar as seguintes funções:
# 
# * <tt>knn()</tt>
# * <tt>svd_sgd_optimizer()</tt>
# * <tt>rmse_mae()</tt>
# * <tt>precision_recall_f1()</tt>
# * <tt>ndcg()</tt>

# %% [markdown]
# ## Parte 1: Leitura e preparação da base de dados
# 
# A base de dados utilizada nesse trabalho foi extraída do site [movielens.org](https://movielens.org). 
# 
# O **MovieLens** é um portal de recomendação de filmes gerenciado pelo grupo de pesquisa GroupLens, da Universidade de Minnesota. Diversas variações da base de dados encontram-se hospedadas no [site do grupo](https://grouplens.org/datasets/movielens/). Ao longo da disciplina, utilizaremos a **ml-latest-small**, versão reduzida, destinada para estudo.
# 
# A base original possui diversos arquivos diferentes, seguindo um determinado padrão. Nem todos serão necessários para o trabalho. Iremos então formatar a base, selecionando apenas as informações relevantes. Adicionalmente, usaremos apenas 20% da base, com o intuito de reduzir o tempo de execução dos experimentos presentes no notebook.
# 
# Um código para preprocessamento e formatação da base encontra-se na função `format_movielens_dataset()`, presente no arquivo `recsys_utils.py` enviado junto com este trabalho. Iremos executá-lo.

# %%
# -*- coding: utf-8 -*-

# Caminho dos arquivos
FILES_DIRECTORY = "ml-100k"

if __name__ == '__main__':
    from recsys_utils import format_movielens_dataset

    format_movielens_dataset(raw_dataset_folder=FILES_DIRECTORY, sampling_rate=0.2)
    print()

# %% [markdown]
# É fortemente encorajado que você olhe o código para entender o que foi realizado. Após sua chamada, **3** novos arquivos foram criados, dentro da pasta <tt>datasets</tt>. São eles:
# * **users.csv**: contendo informações sobre os usuários do sistema;
# * **items.csv**: contendo informações sobre os filmes do sistema;
# * **interactions.csv**: contendo todas as notas que os usuários deram aos filmes.
# 
# Vamos carregar os arquivos como objetos `pandas.DataFrame` e visualizá-los.

# %%
import csv
import pandas as pd

if __name__ == '__main__': 
    # Carrega bases de dados
    users = pd.read_csv('dataset/users.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, encoding='latin-1', header=0, index_col=None)
    items = pd.read_csv('dataset/items.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, encoding='latin-1', header=0, index_col=None)
    interactions = pd.read_csv('dataset/interactions.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, encoding='latin-1', header=0, index_col=None)

    # Preenche generos nulos
    items['genero'] = items['genero'].fillna('')

    # Exibe as 5 primeiras linhas de cada arquivo
    print('\nUsuários:')
    display(users.head(5))
    print('\nItens:')
    display(items.head(5))
    print('\nInterações:')
    display(interactions.head(5))

    # Faz a contagem de usuarios, itens e interacoes
    n_users = len(users)
    n_items = len(items)
    n_interactions = len(interactions)
    print('Temos {} usuários'.format(len(users)))
    print('Temos {} itens'.format(len(items)))
    print('Temos {} interações\n'.format(len(interactions)))

# %% [markdown]
# Com a base de dados carregada em memória, podemos ir para a segunda etapa.

# %% [markdown]
# ---
# ## Parte 2: Divisão da base em Treinamento, Validação e Teste
# 
# Para treinar, ajustar os parâmetros e avaliar nossos modelos, iremos separar a base de dados usando _holdout_, com **80% dos dados para treinamento**, **10% para validação** e **10% para teste**. Para isso, usaremos a função `train_test_split()` da biblioteca `sklearn`. A função só consegue separar a base de dados em duas novas, não conseguindo assim separar em treino, validação e teste com uma única chamada. Iremos então extrair primeiramente as amostras de treinamento, separando o restante entre validação e teste.

# %%
from sklearn.model_selection import train_test_split

if __name__ == '__main__': 
    interactions_train, remaining = train_test_split(
        interactions,
        train_size=0.8,
        test_size=0.2,
        shuffle=False
    )
    interactions_train = interactions_train.copy() # Apenas para evitar mensagens de warning adiante

    interactions_val, interactions_test = train_test_split(
        remaining,
        train_size=0.5,
        test_size=0.5,
        shuffle=False
    )
    interactions_val = interactions_val.copy() # Apenas para evitar mensagens de warning adiante
    interactions_test = interactions_test.copy() # Apenas para evitar mensagens de warning adiante

# %% [markdown]
# ### Remoção do problema de Cold-Start na validação e teste
# 
# Após a separação em treino-validação-teste, pode acontecer de clientes ou itens ficarem presentes apenas nas bases de teste, e ausentes na de treino. Esse é o problema do **cold-start**, ou partida fria, que deve ser abordado por algoritmos específicos (como baseados em conteúdo).
# 
# Como este problema foge do escopo do notebook, vamos remover todos os casos de cold-start:

# %%
if __name__ == '__main__': 
    interactions_val = interactions_val[
        (interactions_val['id_usuario'].isin(interactions_train['id_usuario']))
        &(interactions_val['id_item'].isin(interactions_train['id_item']))
    ].copy()

    interactions_test = interactions_test[
        (interactions_test['id_usuario'].isin(interactions_train['id_usuario']))
        &(interactions_test['id_item'].isin(interactions_train['id_item']))
    ].copy()

    items = items[items['id_item'].isin(interactions_train['id_item'])]
    users = users[users['id_usuario'].isin(interactions_train['id_usuario'])]

    print('Nossa base de TREINAMENTO contém {} interações\n'.format(len(interactions_train)))
    print('Nossa base de VALIDAÇÃO contém {} interações\n'.format(len(interactions_val)))
    print('Nossa base de TESTE contém {} interações\n'.format(len(interactions_test)))

# %% [markdown]
# ### Codifica IDs dos usuários e items para ficar no intervalo de 0 a N
# 
# Para simplificar as implementações seguintes, iremos alterar os IDs dos usuários e itens, para que todos passem a variar no intervalo 0 a "qtde. de usuários" / "qtde. de itens". Com essa alteração, será mais fácil trabalhar com os dados em formato matricial, de forma que uma linhda de índice _i_ corresponda ao item _i_, por exemplo

# %%
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__': 
    item_encoder = LabelEncoder()    
    items.loc[:, 'id_item'] = item_encoder.fit_transform(items['id_item'].values)

    user_encoder = LabelEncoder()
    users.loc[:, 'id_usuario'] = user_encoder.fit_transform(users['id_usuario'].values)

    interactions_train.loc[:, 'id_item'] = item_encoder.transform(interactions_train['id_item'].values)
    interactions_train.loc[:, 'id_usuario'] = user_encoder.transform(interactions_train['id_usuario'].values)

    interactions_val.loc[:, 'id_item'] = item_encoder.transform(interactions_val['id_item'].values)
    interactions_val.loc[:, 'id_usuario'] = user_encoder.transform(interactions_val['id_usuario'].values)

    interactions_test.loc[:, 'id_item'] = item_encoder.transform(interactions_test['id_item'].values)
    interactions_test.loc[:, 'id_usuario'] = user_encoder.transform(interactions_test['id_usuario'].values)

# %% [markdown]
# ---
# ## Parte 3: Implementação dos algoritmos de recomendação
# 
# Serão implementados dois algoritmos diferentes, que utilizam dados explícitos para gerar a recomendação.
# * **K-Vizinhos Mais Próximos (KNN)**, baseado em vizinhança
# * **Decomposição em Valores Singulares (SVD)**, baseado em fatoração de matriz

# %% [markdown]
# ### <center>**KNN**</center>
# 
# Para a implementação do KNN, primeiro iremos converter o DataFrame para uma matriz de interações, e então iremos gerar a recomendação como visto em aula:
# 
# #### **Converter DataFrame para Matriz de Interações**
# 
# Antes de implementar as funções de predição de nota pelo KNN, é necessário que nossos dados sejam representados como uma **Matriz de Interações**, onde cada linha corresponde a um item e cada coluna um usuário (ou o contrário). Com essa matriz, podemos recuperar os vetores de representação dos itens para o restante do algoritmo. Para isso, vamos transformar o dataframe de treino em uma matriz de interações:

# %%
import numpy as np

if __name__ == '__main__': 
    interactions_matrix = np.zeros((n_items, n_users))
    interactions_matrix[interactions_train['id_item'], interactions_train['id_usuario']] = interactions_train['nota'].values
    print(interactions_matrix)

# %% [markdown]
# #### **Predição de nota através do KNN**
# 
# Com a matriz de interações, podemos realizar a predição das notas. A predição é composta de algums etapas, que você deverá implementar:
# 
# * Calcular as similaridades entre os itens, o que pode ser feito usando `sklearn.metrics.pairwise.cosine_similarity`
# * Percorrer cada um dos pares usuário-item que se deseja prever uma nota
# * Encontrar os $k$ itens mais similares ao item alvo **dentre os itens que o usuário já avaliou**
# * Calcular a nota através da seguinte fórmula:
# 
# $$R = \frac{\sum_{i=1}^{k}{(y_{u,i} \times s_i)}}{\sum_{i=1}^{k}{s_i}}$$
# 
# Na qual, $y_{u,i}$ corresponde a nota atribuída pelo usuário ao item vizinho e $s_i$ representa a similaridade com o item alvo. Trata-se de uma simples média ponderada.
# 
# Você deverá implementar a função `knn()`, que recebe as notas dadas pelos usuários num formato de matriz de interações, os pares usuário-item que se deseja prever uma nota e um valor de $k$.
# 
# **Observações**
# * Para calcular as similaridades entre item, pode-se utilizar `sklearn.metrics.pairwise.cosine_similarity`
# * No caso de não ser possível calcular uma nota (todos os itens avaliados pelo usuário tem similaridade 0.0 em relação ao item alvo), preencha com o valor padrão **2.5**

# %%
from sklearn.metrics.pairwise import cosine_similarity

def knn(interactions_matrix, user_item_targets, k=10):
    """
    Preve as notas que os usuarios dariam para os itens usando o algoritmo de KNN
    
    ----------- Entrada -----------
       
    interactions_matrix: matriz np.array contendo as representacoes vetoriais de cada item por linha
    
    user_item_targets: np.array contendo os pares usuario-item para prever a nota, no qual a
        primeira coluna corresponde ao ID do usuario e a segunda coluna ao ID do item        
    
    k: numero de itens mais similares ao item alvo para observar
    
    ----------- Saída -----------
    
    item_sims: matriz contendo as similaridades entre items
    
    ratings: np.array de mesmo tamanho de user_item_targets contendo as notas previstas pelo KNN
    """
    
    # Inicializa a matriz de similaridades
    n_items, n_users = interactions_matrix.shape
    item_sims = np.zeros((n_items, n_items))
    
    # Inicializa as notas
    ratings = np.zeros(user_item_targets.shape[0])
    
    # Nota padrao
    DEFAULT_RATING = 2.5
        
    ###########################################################################
    ######################### COMPLETE O CÓDIGO AQUI  #########################
    # Instruções: você deve prever as notas para os pares usuario-item contidos
    #   em user_item_targets
    #
    # Para tal, calcula a similaridade entre os itens usando a funcao da 
    #   biblioteca sklearn cosine_similarity. Em seguida, para cada par
    #   usuario-item, voce deve calcular a nota prevista usando os K itens
    #   mais similares que o usuario ja avaliou

    item_sims = cosine_similarity(interactions_matrix)
    

    for i, (n_users, n_items) in enumerate(user_item_targets):
        # Indices dos itens que o usuario ja avaliou
        rated_items = np.where(interactions_matrix[:, n_users] > 0)[0]
        user_r = interactions_matrix[interactions_matrix[:, n_users] > 0, n_users]
        sim = item_sims[n_items, rated_items]    
        k_idx = np.argsort(sim)[-k:]# Indices dos k itens mais similares q o usuario ja avaliou
        sim_k = sim[k_idx]
        r_k = user_r[k_idx]
        if np.sum(sim_k) > 0:
            ratings[i] = np.dot(r_k, sim_k) / np.sum(sim_k)           
        else:
            ratings[i] = DEFAULT_RATING



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    
    return item_sims, ratings


if __name__ == '__main__': 
    # Verifica a implementacao
    print("Se seu código estiver correto, a nota prevista será 4.2184")
    item_sims, pred_rating = knn(interactions_matrix, np.array([[398,829]]), k=3)
    print("A nota prevista foi {:.4f}\n".format(pred_rating[0]))

    print("Se seu código estiver correto, a nota prevista será 3.4332")
    item_sims, pred_rating = knn(interactions_matrix, np.array([[630,150]]), k=4)
    print("A nota prevista foi {:.4f}\n".format(pred_rating[0]))

    print("Se seu código estiver correto, a nota prevista será 3.8214")
    item_sims, pred_rating = knn(interactions_matrix, np.array([[723,863]]), k=10)
    print("A nota prevista foi {:.4f}\n".format(pred_rating[0]))
    
    print("Se seu código estiver correto, a similaridade entre os itens 0 e 7 será 0.1687")
    print("A similaridade é {:.4f}\n".format(item_sims[0, 7]))
    
    print("Se seu código estiver correto, a similaridade entre os itens 101 e 368 será 0.4586")
    print("A similaridade é {:.4f}\n".format(item_sims[101, 368]))
    
    print("Se seu código estiver correto, a similaridade entre os itens 888 e 2 será 0.3588")
    print("A similaridade é {:.4f}\n".format(item_sims[888, 2]))

# %% [markdown]
# ### <center> **SVD** </center>
# 
# O segundo método será a técnica de fatoração de matrizes chamada de SVD, ou _Singular Value Decomposition_.
# 
# Iremos otimizar o método através da abordagem **Stochastic Gradient Descent**, em que ambos os vetores de fatores latentes são otimizados simultaneamente.
# 
# #### **Treinamento do SVD para ajuste de fatores latentes**
# 
# Você deverá implementar a função `svd_sgd_optimizer()`, que recebe dois vetores de fatores latentes, de usuário e de item, assim como demais parâmetros da otimização e o valor-alvo (nota atribuída), e deve calcular os novos vetores de fatores latentes minimizando o erro, **utilizando regularização**.
# 
# Primeiro, você deve calcular a diferença entre o valor real e a predição:
# 
# $$e_{u,i} = y_{u,i} - p_uq_i^T$$
# 
# Em seguida, você irá atualizar o conteúdo dos vetores de usuário ($p_u$) e item ($q_i$) com base em uma taxa de aprendizado $\alpha$ e um fator de regularização $\lambda$:
# 
# $$p_{u} = p_{u} + \alpha \cdot (e_{u,i} \cdot q_i - \lambda \cdot p_u)$$
# $$q_{i} = q_{i} + \alpha \cdot (e_{u,i} \cdot p_u - \lambda \cdot q_i )$$
# 
# **Importante**: a atualização de $p_u$ e $q_i$ deve ser feita **simultaneamente**. Assim, antes de atualizar o valor de qualquer um deles, você deve já ter calculado o gradiente.

# %%
def svd_sgd_optimizer(pu, qi, alpha_lr, lambda_reg, y):
    """
    Ajusta os valores pu e qi simultaneamente para aproximá-los de y, utilizando regularização
    
    ----------- Entrada -----------
       
    pu e qi: vetores np.array com os fatores latentes do usuário e do item, respectivamente
            - pu possui dimensões 1 x f, onde f é o número de fatores latentes
            - qi possui dimensões 1 x f, onde f é o número de fatores latentes
        
    alpha_lr: número real representando a taxa de aprendizado
    
    lambda_reg: número real representando o fator de regularizacao
    
    y: nota que o usuário atribuiu ao item
    
    ----------- Saída -----------
    
    new_pu: novo vetor de fatores latentes do usuário
    
    new_qi: novo vetor de fatores latentes do item
    """
    
    new_pu = np.zeros(len(pu)) # Inicializa o novo vetor do usuario
    new_qi = np.zeros(len(qi)) # Inicializa o novo vetor do item
                   
    ###########################################################################
    ######################### COMPLETE O CÓDIGO AQUI  #########################
    # Instruções: você deve calcular os novos valores para os vetores de fator
    # latente do item e do usuário
    #
    # Importante: lembre de atualizar os dois vetores simultaneamentes, ou seja,
    # não use o valor alterado de um dos vetores para atualizar o outro

    e = y - np.dot(pu, qi.T)
    new_pu = pu + alpha_lr * (e * qi - lambda_reg * pu)
    new_qi = qi + alpha_lr * (e * pu - lambda_reg * qi)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    
    return new_pu, new_qi


if __name__ == '__main__': 
    # Verifica a implementacao

    # Inicializa vetores U e I aleatoriamente
    U = np.array([
        [ 0.9,  0.6,  0.5], # Usuário 0
        [-0.2,  0.3, -0.5], # Usuário 1
        [ 2.8,  1.6, -0.1], # Usuário 2    
    ])

    I = np.array([
        [ 0.2,  9.3, -8.3], # Item 0
        [-0.4, -2.3,  5.4], # Item 1
        [-1.1, -0.9,  7.6], # Item 2    
    ])

    print("Se seu código estiver correto, o novo vetor do usuário será aprox.: [ 0.7775 -0.0998  2.1411]")
    new_pu, new_qi = svd_sgd_optimizer(pu=U[0], qi=I[1], alpha_lr=0.10, lambda_reg=0.01, y=4.0)
    print("O novo vetor do usuário é {}\n".format(new_pu))

    print("Se seu código estiver correto, o novo vetor do item será aprox.: [-0.66212 -0.64924  7.56924]")
    new_pu, new_qi = svd_sgd_optimizer(pu=U[2], qi=I[2], alpha_lr=0.02, lambda_reg=0.1, y=2.5)
    print("O novo vetor do item é {}\n".format(new_qi))

    print("Se seu código estiver correto, o novo vetor do usuário será aprox.: [-0.59275 -0.0225   2.21825]")
    new_pu, new_qi = svd_sgd_optimizer(pu=U[1], qi=I[2], alpha_lr=0.05, lambda_reg=0.05, y=3.3)
    print("O novo vetor do usuário é {}\n".format(new_pu))

    print("Se seu código estiver correto, o novo vetor do item será aprox.: [-0.36054 -2.25336  5.3657 ]")
    new_pu, new_qi = svd_sgd_optimizer(pu=U[0], qi=I[1], alpha_lr=0.01, lambda_reg=1, y=4.9)
    print("O novo vetor do item é {}\n".format(new_qi))

# %% [markdown]
# #### **Predição de nota através do SVD**
# 
# Em seguida, iremos implementar a função `svd()`, que irá utilizar sua função implementada na célula anterior.
# 
# Esta função irá, ao longo de uma série de iterações, ajustar os fatores latentes de usuário e item para todas as interações e realizar a predição das notas para um conjunto de usuários-itens.

# %%
def svd(interactions_matrix, user_item_targets, n_factors=100, n_epochs=10, alpha_lr=0.01, lambda_reg=0.01, verbose=True):
    
    # Inicializa os fatores latentes aleatoriamente
    np.random.seed(0)
    I = np.random.rand(interactions_matrix.shape[0], n_factors)
    np.random.seed(0)
    U = np.random.rand(interactions_matrix.shape[1], n_factors)
    
    ############################ TREINAMENTO ############################
    
    # Ao longo de N iterações...
    for epc in range(n_epochs):
        # Para cada interação...
        all_interactions = interactions_matrix.nonzero()
        for itr, (item, user) in enumerate(zip(*all_interactions), start=1):
            if itr % 100 == 0 and verbose:
                print("SVD - Iteração: {:02d}/{:02d}  |  Interação: {:05d}/{:05d}".format(epc+1, n_epochs, itr, len(all_interactions[0])), end='\r', flush=True)
            # Recupera a nota
            rating = interactions_matrix[item, user]
            # Calcula novos vetores de fatores latentes
            new_pu, new_qi = svd_sgd_optimizer(U[user], I[item], alpha_lr, lambda_reg, rating)
            # Atualiza vetores de fatores latentes
            U[user] = new_pu
            I[item] = new_qi
        if verbose:
            print("SVD - Iteração: {:02d}/{:02d}  |  Interação: {:05d}/{:05d}".format(epc+1, n_epochs, itr, len(all_interactions[0])), end='\r', flush=True)
       
    ############################ PREDICAO DA NOTA ############################
    
    all_ratings = np.clip(np.dot(U, I.T), a_min=1, a_max=5) # Delimita a predição dentro do intervalo de notas
    ratings = all_ratings[user_item_targets[:,0], user_item_targets[:,1]]
    
    return ratings

# %% [markdown]
# ---
# ## Parte 4: Métricas de avaliação
# 
# Com nossos algoritmos implementados, estamos prontos para recomendar. Entretanto, como podemos saber qual algoritmo foi melhor? Para isso, são usadas métricas de avaliação.
# 
# Iremos implementar duas famílias diferentes de métricas avaliativas: aquelas que medem a qualidade de um algoritmo para prever notas, e as que medem a qualidade de um algoritmo em recomendar uma lista de itens de forma ordenada.
# 
# ### <center> **Predição de Nota** </center>
# 
# As duas métricas mais comuns quando temos algoritmos que prevem uma nota são o **Erro Médio Absoluto (MAE)** e a **Raiz do Erro Médio Quadrático (RMSE)**. Ambos são medidas de erro, assim, quanto **menor** o valor, **melhor** o método.
# 
# Você deverá implementar a função `rmse_mae()`, responsável por calcular cada uma das métricas usando as seguintes fórmulas:
# 
# $$\text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{n} \times \sum_{i=1}^{n}{(y_i - \hat{y}_i)^2}}$$
# 
# $$\text{MAE}(y, \hat{y}) = \frac{1}{n} \times \sum_{i=1}^{n}{\lvert y_i - \hat{y}_i \rvert}$$
# 
# A função irá receber dois vetores como entrada: um com as notas reais ($y$, na variável `real`) e outro com as notas previstas ($\hat{y}$, na variável `pred`). O retorno será dois números reais, um para cada métrica, variando de $0$ a $\infty$.

# %%
import numpy as np

def rmse_mae(real, pred):
    """
    Calcula a Raiz do Erro Médio Quadrático e o Erro Médio Absoluto entre os valores de real e pred
    
    ----------- Entrada -----------
       
    real: vetor np.array com as notas reais
    
    pred: vetor np.array com as notas previstas
    
    ----------- Saída -----------
            
    rmse_score: Raiz do Erro Médio Quadrático da predição
    
    mae_score: Erro Médio Absoluto da predição
    """
    
    mae_score = 0.0 # Inicializa o MAE
    rmse_score = 0.0 # Inicializa o RMSE
    
    ###########################################################################
    ######################### COMPLETE O CÓDIGO AQUI  #########################
    # Instruções: você deve calcular o MAE entre as notas reais, em real, e
    # previstas, em pred

    mae_score = np.mean(np.abs(real - pred))
    rmse_score = np.sqrt(np.mean((real - pred)**2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
    
    return rmse_score, mae_score

if __name__ == '__main__': 
    # Verifica a implementacao
    print("Se seu código estiver correto, o MAE será 1.333 e o RMSE será 1.6330")
    rmse_score, mae_score = rmse_mae(np.array([1, 5, 3]), np.array([1, 3, 1]))
    print("O MAE é {:.4f} e o RMSE é {:.4f}\n".format(mae_score, rmse_score))

    print("Se seu código estiver correto, o MAE será 2.5000 e o RMSE será 2.7386")
    rmse_score, mae_score = rmse_mae(np.array([5, 5, 5, 5]), np.array([1, 2, 3, 4]))
    print("O MAE é {:.4f} e o RMSE é {:.4f}\n".format(mae_score, rmse_score))

    print("Se seu código estiver correto, o MAE será 0.6200 e o RMSE será 0.8379")
    rmse_score, mae_score = rmse_mae(np.array([1.5, 2.6, 4.9, 3.4, 2.8]), np.array([1.1, 4.3, 5.0, 3.1, 2.2]))
    print("O MAE é {:.4f} e o RMSE é {:.4f}\n".format(mae_score, rmse_score))

# %% [markdown]
# ### <center> **Ranqueamento Top-N** </center>
# 
# Das métricas que avaliam a qualidade do ranqueamento top-N de um algoritmo, implementaremos 4 delas: Precisão, Revocação, F-Medida e NDCG, sendo que a última será implementada por você!
# 
# Todas as métricas consumirão um dataframe de recomendações que contém três colunas: usuário-alvo, item recomendado e o rank da recomendação (que irá variar de 1 a N). Para implementarmos as métricas, iremos gerar uma base falsa de interações para usar como teste, e uma base falsa de recomendações para usar como a recomendação top-N.

# %%
if __name__ == '__main__': 
    # Monta a base de interações de teste
    fake_real = pd.DataFrame(
        [[0, 0], [0, 1], [1, 1], [1, 10], [1, 100], [2, 11], [2, 22], [3, 100], [4, 0], [4, 20], [4, 100], [4, 400]],
        columns=['id_usuario', 'id_item']
    )
    print("Interações de teste")
    display(fake_real)

    # Monta as recomendações
    fake_pred = pd.DataFrame(
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], 
         [1, 0, 1], [1, 10, 2], [1, 20, 3],
         [2, 10, 1], [2, 20, 2], [2, 22, 3], 
         [3, 1, 1], [3, 11, 2], [3, 400, 3],
         [4, 0, 1], [4, 20, 2], [4, 400, 3]],
        columns=['id_usuario', 'id_item', 'rank']
    )
    print("Recomendações")
    display(fake_pred)

# %% [markdown]
# ### **Precisão, Revocação e F-Medida**
# 
# Para avaliar a qualidade dos itens selecionados por cada algoritmo, podemos comparar o conteúdo das listas top-$N$ com os itens consumidos pelos usuários numa base de teste, calculando métricas avaliativas.
# 
# A precisão busca calcular quantos acertos houveram dentro de tudo que foi recomendado, sendo descrita pela fórmula:
# 
# $$\text{Prec} = \frac{\text{Qtde de recomendações corretas}}{\text{Qtde de itens recomendados}} = \frac{\text{Qtde de recomendações corretas}}{\text{Qtde de usuários} \times N}$$
# 
# A revocação busca verificar quanto foi acertado de tudo aquilo que realmente poderia ser acertado. É calculada através da fórmula:
# 
# $$\text{Rec} = \frac{\text{Qtde de recomendações corretas}}{\text{Qtde de itens consumidos pelos usuários}} = \frac{\text{Qtde de recomendações corretas}}{\text{Tamanho da base de teste}}$$
# 
# Por fim, a F-Medida é uma forma de equilibrar a Precisão e a Revocação em uma métrica única. É calculada como:
# 
# $$\text{F1} = 2 \times \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}}$$
# 
# Você deve implementar a função `precision_recall_f1()`, que recebe como entrada um conjunto de interações real e um previsto, e retorna a precisão, a revocação e a f-medida.

# %%
import pandas as pd

def precision_recall_f1(real, pred):
    """
    Calcula a precisão, revocação e f-medida da recomendação
    
    ----------- Entrada -----------
       
    real: DataFrame com as interações reais, de forma similar a interactions_test
    
    pred: DataFrame com as recomendações, no formato (usuario, item, rank)
    
    ----------- Saída -----------
    
    precision_score: Número real contendo a precisão da recomendação
    
    recall_score: Número real contendo a revocação da recomendação
    
    f1_score: Número real contendo a f-medida da recomendação, que deverá ser 0 se precision e recall também forem 0
    
    """
    
    precision_score = 0.0 # Inicializa a precisão
    recall_score = 0.0 # Inicializa a revocação
    f1_score = 0.0 # Inicializa a f-medida
    
    # encontra as recomendações feitas corretamente
    hits = real.merge(pred, on=['id_usuario', 'id_item'], how='inner')

    ###########################################################################
    ######################### COMPLETE O CÓDIGO AQUI  #########################
    # Instruções: você deve calcular a precisão, a revocação e a f-medida
    #  da recomendação

    precision_score = len(hits) / len(pred)
    recall_score = len(hits) / len(real)
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    ###########################################################################
    
    return precision_score, recall_score, f1_score

    
if __name__ == '__main__': 
    # Verifica a implementacao
    prec, rec, f1 = precision_recall_f1(fake_real, fake_pred)

    print("Se seu código estiver correto, a precisão da recomendação será aprox.: 0.4667")
    print("A precisão da recomendação é: {:.4f}\n".format(prec))

    print("Se seu código estiver correto, a revocação da recomendação será aprox.: 0.5833")
    print("A revocação da recomendação é: {:.4f}\n".format(rec))

    print("Se seu código estiver correto, a f-medida da recomendação será aprox.: 0.5185")
    print("A f-medida da recomendação é: {:.4f}\n".format(f1))

# %% [markdown]
# ### **Ganho Cumulativo Descontado Normalizado (NDCG)**
# 
# O NDCG é uma métrica que busca avaliar a qualidade do ordenamento, ponderando as recomendações corretas de acordo com a sua posição no rank e usando um decaimento logaritmico.
# 
# É calculado como:
# 
# $$NDCG = \frac{DCG}{IDCG}$$
# 
# Onde DCG representa o ganho não normalizado, calculado por:
# 
# $$DCG = \sum_{u \in U}{\sum_{i \in R_u}{\frac{rel(i, u)}{log_2(rank_i + 1)}}}$$
# 
# Onde $U$ é o conjunto de usuários, $R_u$ são os itens recomendados para o usuário, $rank_i$ é a posição do item na recomendação e $rel(i, u)$ é
# 
# $$rel(i, u) = 0 \text{ se o usuário } u \text{ NÃO interagiu com o item } i \text{, e } 1 \text{ caso contrário}$$
# 
# Já IDCG representa o DCG ideal, ou seja, aquele que seria obtido numa recomendação perfeita. É dado por:
# 
# $$IDCG = \sum_{u \in U}{\sum_{k = 1}^{T_{u_N}}{\frac{1}{log_2(rank_i + 1)}}}$$
# 
# Idêntico ao DCG, mas com a suposição que todos os itens foram encontrados e ordenados corretamente. 
# 
# **IMPORTANTE**: por procurar o ganho da recomendação ideal, é importante que o IDCG considere apenas as primeiras $N$ posições. Assim, se um usuário consumiu mais itens que $N$ na base de teste, o IDCG deve ser calculado apenas para $N$ itens, descartando os demais, assim é possível vermos como seria uma recomendação perfeita.
# 
# Você deve implementar a função `ndgc()`, que recebe como entrada um conjunto de interações real, um previsto (tal como a função `precision_recall_f1`) e o $N$ da recomendação, e retorna o DCG, o IDCG e o NDCG.
# 
# **Dica**: observe na função `precision_recall_f1()` como foi feita a descoberta das recomendações corretas. Isso pode te ajudar na implementação da função

# %%
def ndcg(real, pred, top_n):
    """
    Calcula o DCG, IDCG e NDCG da recomendação
    
    ----------- Entrada -----------
       
    real: DataFrame com as interações reais, de forma similar a interactions_test
    
    pred: DataFrame com as recomendações, no formato (usuario, item, rank)
    
    top_n: valor de N para a recomendação top-N
    
    ----------- Saída -----------
    
    dcg_score: Número real contendo o ganho cumulativo descontado (DCG) da recomendação
    
    idcg_score: Número real contendo o ganho cumulativo descontado ideal (IDCG) da recomendação
    
    ndcg_score: Número real contendo o ganho cumulativo descontado normalizado (NDCG) da recomendação
    
    """
    
    dcg_score = 0.0   # Inicializa o DCG
    idcg_score = 0.0  # Inicializa o IDCG
    ndcg_score = 0.0  # Inicializa o NDCG
    
    ###########################################################################
    ######################### COMPLETE O CÓDIGO AQUI  #########################
    # Instruções: você deve calcular o DCG, o IDCG e o NDCG da recomendação

    for user in real['id_usuario'].unique():
        items_real = real[real['id_usuario'] == user]['id_item'].values
        items_pred = pred[pred['id_usuario'] == user]['id_item'].values  #items recomendados para o usuário
        rel = np.isin(items_pred, items_real)
        dcg = np.sum(rel / np.log2(np.arange(2, rel.size + 2))) 
        dcg_score += dcg
        for i in range(1, top_n + 1):
            if i <= len(items_real):
                idcg_score += 1 / np.log2(i + 1)
            else:
                break
        
    ndcg_score = dcg_score / idcg_score


            
    return dcg_score, idcg_score, ndcg_score
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    ###########################################################################
    
    return dcg_score, idcg_score, ndcg_score


if __name__ == '__main__': 
    # Verifica a implementacao
    dcg_score, idcg_score, ndcg_score = ndcg(fake_real, fake_pred, top_n=3)

    print("Se seu código estiver correto, o DCG da recomendação será aprox.: 4.8928")
    print("O DCG da recomendação é: {:.4f}\n".format(dcg_score))

    print("Se seu código estiver correto, o IDCG da recomendação será aprox.: 8.5237")
    print("O IDCG da recomendação é: {:.4f}\n".format(idcg_score))

    print("Se seu código estiver correto, o NDCG da recomendação será aprox.: 0.5740")
    print("O NDCG da recomendação é: {:.4f}\n".format(ndcg_score))

# %% [markdown]
# ---
# ## Parte 5: Ajuste de parâmetros
# 
# Nesta etapa, iremos testar diversos valores para os hiperparâmetros dos modelos, para descobrir, através de uma busca em grande sobre a validação, qual a melhor combinação de parâmetros. Os parâmetros selecionados serão aqueles que minimizarem a métrica RMSE.

# %%
if __name__ == '__main__': 
    # Recupera os pares usuario-item alvo e as notas reais
    user_item_targets = interactions_val[['id_usuario', 'id_item']].values
    real_ratings = interactions_val['nota'].values

    # --- KNN
    print("---------- KNN ----------")
    knn_best_params, knn_best_rmse = None, np.inf
    for k in [3, 5, 10, 15]:
        _, pred_ratings = knn(interactions_matrix, user_item_targets, k=k)
        pred_rmse, _ = rmse_mae(real_ratings, pred_ratings)
        print('\tk = {}  |  RMSE: {:.5f}'.format(k, pred_rmse))
        if pred_rmse < knn_best_rmse:        
            knn_best_rmse = pred_rmse
            knn_best_params = {'k': k}
    print("Melhores parâmetros encontrados para o KNN: {}\n".format(knn_best_params))

    # --- SVD
    print("---------- SVD ----------")
    svd_best_params, svd_best_rmse = None, np.inf
    for n_factors in [50, 100]:
        for alpha_lr in [0.01, 0.001]:
            for lambda_reg in [0.01, 0.001]:        
                pred_ratings = svd(interactions_matrix, user_item_targets, n_factors=n_factors, alpha_lr=alpha_lr, lambda_reg=lambda_reg, verbose=False)
                rmse_score, _ = rmse_mae(real_ratings, pred_ratings)
                print('\tn_factors = {}  |  alpha_lr = {}  |  lambda_reg = {} |  RMSE: {:.5f}'.format(n_factors, alpha_lr, lambda_reg, pred_rmse))            
                if pred_rmse < svd_best_rmse:
                    svd_best_rmse = pred_rmse
                    svd_best_params = {'n_factors': n_factors, 'alpha_lr': alpha_lr, 'lambda_reg': lambda_reg}
    print("Melhores parâmetros encontrados para o SVD: {}\n".format(svd_best_params))

# %% [markdown]
# ---
# ## Parte 6: Avaliação dos algoritmos de recomendação
# 
# Finalmente, podemos executar nossos algoritmos sobre a base de teste e verificar o resultado final.

# %% [markdown]
# #### **Junta bases de treinamento e validação**
# 
# Para enriquecer o treinamento, iremos juntar nossa base de validação com a de treinamento.

# %%
if __name__ == '__main__': 
    interactions_all = pd.concat([interactions_train, interactions_val]).copy()

    interactions_matrix = np.zeros((n_items, n_users))
    interactions_matrix[interactions_all['id_item'], interactions_all['id_usuario']] = interactions_all['nota'].values

# %% [markdown]
# #### **Tarefa de predição de nota**
# 
# Nesta tarefa, os algoritmos irão prever as notas para um conjunto de pares usuário-item, tendo suas predições avaliadas pelo MAE e o RMSE.

# %%
if __name__ == '__main__': 
    # Recupera pares usuario-item
    rating_pred_user_item_targets = interactions_test[['id_usuario', 'id_item']].values
    real_rating_pred = interactions_test['nota'].values

    # Preve as notas
    print('Prevendo as notas...')
    _, knn_rating_pred = knn(interactions_matrix, rating_pred_user_item_targets, **knn_best_params)
    svd_rating_pred = svd(interactions_matrix, rating_pred_user_item_targets, **svd_best_params)

    # Calcula os resultados
    print('\nCalculando métricas...')
    knn_rmse, knn_mae = rmse_mae(knn_rating_pred, real_rating_pred)
    svd_rmse, svd_mae = rmse_mae(svd_rating_pred, real_rating_pred)

    print('OK!')

# %% [markdown]
# #### **Tarefa de ranqueamento top-N**
# 
# Para recomendar utilizando os algoritmos implementados, teríamos que prever as notas que os usuários-alvo dariam para TODOS os itens não consumidos, o que pode ser extremamente custoso. Em sistemas reais, os algoritmos estariam implementados de forma mais otimizada, além de outras estratégias que poderiam ser adotadas, como uma pré-etapa de clusterização ou uso de recomendadores específicos.
# 
# Para que o notebook rode em tempo viável e não demande um computador com alta disponibilidade de recursos, iremos gerar uma ranking top-20 apenas para os 10 usuários na base de teste que mais interagiram com itens, avaliando através da Precisão, Revocação, F-Medida e NDCG.

# %%
if __name__ == '__main__': 
    # Recupera os usuarios-alvo
    NUM_TOP_N_USERS = 10
    top_n_users = interactions_test['id_usuario'].value_counts().sort_values(ascending=False).index.values[:NUM_TOP_N_USERS]

    # Monta um dataframe com TODOS os pares usuario-item usando os usuarios-alvo
    top_n_user_item_targets = pd.DataFrame([], columns=['id_usuario', 'id_item']).astype(int)
    for tnu in top_n_users:    
        not_consumed_items = np.where(interactions_matrix[:, tnu]==0)[0]

        top_n_user_df = pd.DataFrame(
            np.vstack([np.repeat(tnu, len(not_consumed_items)), not_consumed_items]).T,
            columns=['id_usuario', 'id_item']
        ).astype(int)    

        top_n_user_item_targets = pd.concat([top_n_user_item_targets, top_n_user_df])

    # Preve as notas atribuidas pelos usuarios para todos os itens nao consumidos
    print('Prevendo as notas...')
    _, knn_ratings = knn(interactions_matrix, top_n_user_item_targets[['id_usuario', 'id_item']].values)
    svd_ratings = svd(interactions_matrix, top_n_user_item_targets[['id_usuario', 'id_item']].values)
    top_n_user_item_targets['nota-knn'] = knn_ratings
    top_n_user_item_targets['nota-svd'] = svd_ratings

    # Gera a recomendacao top-20
    print('\nMontando listas top-N...')
    TOP_N = 20

    knn_top_n = top_n_user_item_targets.sort_values('nota-knn', ascending=False).groupby('id_usuario').head(TOP_N)[['id_usuario', 'id_item']].sort_values('id_usuario')
    knn_top_n['rank'] = np.tile(np.arange(1, TOP_N+1), NUM_TOP_N_USERS)

    svd_top_n = top_n_user_item_targets.sort_values('nota-svd', ascending=False).groupby('id_usuario').head(TOP_N)[['id_usuario', 'id_item']].sort_values('id_usuario')
    svd_top_n['rank'] = np.tile(np.arange(1, TOP_N+1), NUM_TOP_N_USERS)

    real_top_n = interactions_test[['id_usuario', 'id_item']] # Interacoes de teste

    # Calcula as metricas
    print('Calculando métricas...')
    knn_prec, knn_rec, knn_f1 = precision_recall_f1(real_top_n, knn_top_n)
    _, _, knn_ndcg = ndcg(real_top_n, knn_top_n, TOP_N)

    svd_prec, svd_rec, svd_f1 = precision_recall_f1(real_top_n, svd_top_n)
    _, _, svd_ndcg = ndcg(real_top_n, svd_top_n, TOP_N)

    print('OK!')

# %% [markdown]
# #### **Resultados finais**
# 
# Os resultados obtidos são bastante interessantes. Ainda que em aplicações reais o SVD geralmente obtenha resultados superiores ao KNN, neste experimento, o algoritmo baseado em vizinhança se saiu superior a fatoração de matriz em todas as métricas, o que mostra que está sendo melhor em aproximar as notas relacionadas aos itens da base de teste e ainda aprender uma boa ordem para os itens que o usuário gostaria de consumir. Ainda assim, não podemos afirmar que um método é superior ao outro porque estamos num cenário pouco condizente com a realidade. Fizemos muitas amostragens dos dados e temos uma base com poucos clientes e baixa significância estatística.

# %%
if __name__ == '__main__': 
    results = pd.DataFrame([
        [knn_mae, svd_mae],
        [knn_rmse, svd_rmse],
        [knn_prec, svd_prec],
        [knn_rec, svd_rec],
        [knn_f1, svd_f1],
        [knn_ndcg, svd_ndcg]
    ], columns=['KNN', 'SVD'], index=['MAE', 'RMSE', 'Prec', 'Rec', 'F1', 'NDCG'])
    display(results)

# %%



