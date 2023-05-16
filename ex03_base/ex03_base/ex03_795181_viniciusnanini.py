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
# <center><i><b>
# Atenção: não são autorizadas cópias, divulgações ou qualquer tipo de uso deste material sem o consentimento prévio dos autores.
# </center></i></b>
# <br/>

# %% [markdown]
# # <center>Exercício - Naive Bayes</center>
# 
# Neste exercício, você implementará o método Naive Bayes e verá como ele utiliza os dados para fazer classificações de amostras não vistas. Antes de começar este exercício, é recomendável que você revise os conceitos apresentados em aula.
# 
# ### Descrição do problema
# Rodonildo é um jogador nato de *League of Legends*, um jogo de estratégia que envolve a batalha entre dois times (para maiores detalhes, consulte http://br.leagueoflegends.com/ ), e esteve coletando dados nas partidas em que jogou. O objetivo de Rodonildo é prever o vencedor de uma determinada batalha a partir de algumas informações. Na coleta de dados que Rodonildo fez, ele utilizou amostras compostas pelos 5 atributos binários (*1 = sim* e *0 = não*) a seguir:
# 
# 1. *primeiroAbate*: indica se a primeira morte do jogo foi realizada pelo time de Rodonildo;
# 2. *primeiraTorre*: indica se a primeira torre destruída do jogo foi derrubada pelo time de Rodonildo (Figura 1a);
# 3. *primeiroInibidor*: indica se o primeiro inibidor destruído do jogo foi derrubado pelo time de Rodonildo (Figura 1b);
# 4. *primeiroDragao*: indica se o personagem Dragão foi abatido primeiro pelo time de Rodonildo (Figura 1c);
# 4. *primeiroBaron*: indica se o personagem Baron foi abatido primeiro pelo time de Rodonildo (Figura 1d).
# 
# <div style="display:inline-block;">
#     <div>
#     <div style="padding: 5px; float: left;">
#         <img src="figs/turret.png" style="height:180px;"/>
#         <center><em>(a) Torre</em></center>
#     </div>
#     <div style="padding: 5px; float: left;">
#         <img src="figs/inhibitor.png"  style="height:180px;"/> 
#         <center><em>(b) Inibidor</em></center>
#     </div>
#     <div style="padding: 5px; float: left;">
#         <img src="figs/dragon.png"  style="height:180px;"/>
#         <center><em>(c) Dragão</em></center>
#     </div>
#     <div style="padding: 5px; float: left;">
#         <img src="figs/baron.png"  style="height:180px;"/>
#         <center><em>(d) Baron</em></center>
#     </div>
#     </div>
#     <center><em>Figura 1. Objetos e Criaturas de League of Legends.</em></center>
# </div> 
# 
# Por exemplo, a amostra $x = [0, 0, 1, 1, 0]$ e $y = 0$  representa um jogo no qual o time de Rodonildo destruiu primeiro um inibidor inimigo e derrotou o dragão antes da equipe inimiga. Por sua vez, a equipe adversária fez o primeiro abate do jogo, destruiu a primeira torre e derrotou o Baron. Essa partida foi vencida pela equipe adversária.
# 
# Após longo período de coleta de dados, Rodonildo precisa da sua ajuda para prever o resultado de outras partidas utilizando as informações armazenadas. A sua função é implementar o classificador Naive Bayes para predizer qual será o resultado das próximas partidas de Rodonildo, condicionado aos valores dos atributos.
# 
# Instruções
# ----------
# 
# Este arquivo contém o código que auxiliará no desenvolvimento do exercício. Você precisará completar as seguintes funções:
# 
# * calcularProbabilidades()
# * classificacao()
# * text2features()
# * calcularProbabilidades_Laplace()
# * classificacao_texto()
# 
# Você não poderá criar nenhuma outra função. Apenas altere as rotinas fornecidas.

# %% [markdown]
# ## Parte 1: Calcular as probabilidades
# 
# Nesta etapa, você precisará implementar a função *calcularProbabilidades()*. Esta função retornará os vetores com as probabilidades de cada atributo para as classes.

# %% [markdown]
# Primeiro, vamos carregar a base de dados com as partidas jogadas pelo Rodonildo.

# %%
# -*- coding: utf-8 -*-

# Caminho dos arquivos
FILES_DIRECTORY = "dados"

import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
import os #importa a biblioteca para tarefas relacionadas ao sistema operacional

if __name__ == '__main__':
    # Importa o arquivo e guarda em um dataframe do Pandas
    df_dataset = pd.read_csv( os.path.join(FILES_DIRECTORY, 'dados.csv'), sep=',', index_col=None)

    print('Dados carregados com sucesso!')

# %% [markdown]
# Agora, vamos dar uma olhada nas 6 primeiras amostras da base de dados.

# %%
if __name__ == '__main__':
    # vamos usar a função display para imprimir o dataframe, pois deixa mais bonito. 
    # Mas, também poderíamos ter usado a função print: print(df_dataset.head(n=6))
    display(df_dataset.head(n=6))

# %% [markdown]
# Vamos guardar os dados dentro de uma matriz e as classes dentro de um vetor.

# %%
if __name__ == '__main__':
    # pega os valores das n-1 primeiras colunas e guarda em uma matrix X
    X = df_dataset.iloc[:, 0:-1].values 

    # pega os valores da última coluna e guarda em um vetor Y
    Y = df_dataset.iloc[:, -1].values 

    # imprime as 5 primeiras linhas da matriz X
    display('X:', X[0:5,:])

    # imprime os 5 primeiros valores de Y
    print('Y:', Y[0:5])

# %% [markdown]
# Vamos calcular qual a probabilidade de ocorrência de cada classe.

# %%
if __name__ == '__main__':
    # Probabilidade das Classes
    pVitoria = sum(Y==1)/len(Y) 
    pDerrota = sum(Y==0)/len(Y)

    print('Probabilidade da classe ser 1 (vitória): %1.2f%%' %(pVitoria*100))
    print('Probabilidade da classe ser 0 (derrota): %1.2f%%' %(pDerrota*100))

# %% [markdown]
# Agora, crie a função que irá calcular as probabilidades de ocorrência de cada atributo em cada classe.
# 
# **Obs.** Todos os atributos deste problema de classificação possuem apenas dois valores possíveis (1 \[a ação representada pelo atributo foi tomada pelo time do Rodonildo\] ou 0 [a ação representada pelo atributo foi tomada pelo time adversário]). Portanto, na função abaixo você deverá calcular apenas a probabilidade do atributo possuir valor 1. Posteriormente, na função de classificação, basta considerar que a probabilidade de um determinado atributo possuir valor 0 é complementar à probabilidade do atributo possui valor 1. 

# %%
def calcularProbabilidades(X, Y):
    """
    Computa a probabilidade de ocorrencia de cada 
    atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
    (qtde de atributos), um para cada classe.
    
    """
    
    #  inicializa os vetores de probabilidades
    pAtrVitoria = np.zeros(X.shape[1])
    pAtrDerrota = np.zeros(X.shape[1])

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: Complete o codigo para encontrar a probabilidade de
    #                ocorrencia de um atributo para uma determinada classe.
    #                Ex.: para a classe 1 (vitoria), devera ser computada um
    #                vetor pAtrVitoria (n x 1) contendo n valores:
    #                P(Atributo1=1|Classe=1), ..., P(Atributo5=1|Classe=1), e o
    #                mesmo para a classe 0 (derrota):
    #                P(Atributo1=1|Classe=0), ..., P(Atributo5=1|Classe=0).
    # 

    #guardaremos todos os indices de Y nos quais ele tiver resultado igual a 1
    #Com isso todo X que tiver do indice indicado que for igual a 1 sera somado e divido pela soma das classes igual a 1
    pAtrVitoria = ((X[np.where(Y==1)]==1).sum(axis=0))/sum(Y==1)

    #guardaremos todos os indices de Y nos quais ele tiver resultado igual a 0
    #Com isso todo X que tiver do indice indicado que for igual a 1 sera somado e divido pela soma das classes igual a 0
    pAtrDerrota = ((X[np.where(Y==0)]==1).sum(axis=0))/sum(Y==0)
    
    
    
    
    
    
    
    
    
    
       
    ##########################################################################

    return pAtrVitoria, pAtrDerrota

if __name__ == '__main__':
    pAtrVitoria, pAtrDerrota = calcularProbabilidades(X,Y)

    print('A probabilidade esperada para P(PrimeiroAbate=1|Classe=1) = %.2f%%' %52.96)
    print('\nEssa mesma probabilidade calculada no seu codigo foi = %.2f%%' %(pAtrVitoria[0]*100))

# %% [markdown]
# ## Parte 2: Classificação da própria base usando o método Naive Bayes
# 
# Nesta etapa, é realizada a classificação das amostras com base nas probabilidades encontradas no passo anterior. A classificação é realizada verificando se a amostra em questão tem maior probabilidade de pertencer à classe 1 ou à classe 0. Para calcular a probabilidade de uma amostra pertencer a uma determinada classe, é necessário utilizar as probabilidades de ocorrências de atributos previamente computadas. O cálculo pode ser expresso como:
# 
# $$ P(y_j|\vec{x}) = \hat{P}(y_{j}) \prod_{x_i \in \vec{x}} \hat{P}(x_{i} | y_{j}) $$
# 
# Portanto, a probabilidade de uma amostra $\vec{x}$ pertencer a uma classe $j$ é obtida a partir da probabilidade geral da classe $j$ ($\hat{P}(y_{j})$) multiplicada pelo produtório da probabilidade de ocorrência de cada atributo $x_i$ com relação a classe $j$ ($\hat{P}(x_{i} | y_{j})$).
# 
# Se a rotina de classificação estiver correta, espera-se que a acurácia obtida ao classificar a própria base de amostras de jogos que Ronildo participou seja aproximadamente 76,60%. 
# 
# Você deverá completar a função **Classificacao()**.
# 
# **Dica:**
# * Neste problema de classificação, o valor 0 presente em cada atributo deve ser levado em consideração, pois ele não significa ausência de valor. Ele significa que a ação representada pelo atributo foi tomada pelo time adversário. 

# %%
def classificacao(x,pVitoria,pDerrota,pAtrVitoria,pAtrDerrota):
    """
    Classifica se a entrada x pertence a classe 0 ou 1 usando
    as probabilidades extraidas da base de treinamento. Essa funcao 
    estima a predicao de x atraves da maior probabilidade da amostra  
    pertencer a classe 1 ou 0. Tambem retorna as probabilidades condicionais
    de vitoria e derrota, respectivamente.
    
    """

    #  inicializa a classe e as probabilidades condicionais
    classe = 0
    probVitoria= 0
    probDerrota = 0

    

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: Complete o codigo para estimar a classificacao da amostra
    #                usando as probabilidades extraidas da base de treinamento.
    #                Voce precisara encontrar as probabilidades Bayesianas: 
    #                    . probVitoria -> p(classe=1|x)
    #                    . probDerrota -> p(classe=0|x) 
    #                Depois, você deve selecionar a maior delas. 
    # 
    probVitoria = pVitoria;
    probDerrota = pDerrota;
    for i in range(len(x)):
        #calculando a probabilidade de uma amostra pertencer a uma determinada classe, caso o atributo for igual a 0 subtraimos 1 de yi
        probVitoria = probVitoria * pAtrVitoria[i] if x[i]==1 else probVitoria * (1-pAtrVitoria[i])
        probDerrota = probDerrota * pAtrDerrota[i] if x[i]==1 else probDerrota * (1-pAtrDerrota[i])

    #valor dominante
    if probVitoria > probDerrota:
        classe = 1
    else:
        classe = 0

    ########################################################################## 

    return classe, probVitoria, probDerrota 


if __name__ == '__main__':
    resultados = np.zeros( X.shape[0] )

    for i in range(X.shape[0]):
        resultados[i], probVitoria, probDerrota = classificacao( X[i,:],pVitoria,pDerrota,pAtrVitoria,pAtrDerrota )

    # calcular acuracia
    acuracia = np.sum(resultados==Y)/len(Y)

    print('\n\nAcuracia esperada para essa base = %.2f%%\n' %76.60)
    print('Acuracia obtida pelo seu classificador foi = %.2f%%\n' %( acuracia*100 ) )

# %% [markdown]
# ## Parte 3: Predizendo a classe de novos dados
# 
# Já que toda a etapa de treinamento e classificação está concluída, o último passo é permitir que novas amostras sejam classificadas. 

# %%
if __name__ == '__main__':
    x1_novo = np.array([0,0,0,1,1])

    classe, probVitoria, probDerrota = classificacao( x1_novo,pVitoria,pDerrota,pAtrVitoria,pAtrDerrota )

    if classe ==1:
        print('\n>>> Predicao = Vitoria!')       
    else:
        print('\n>>> Predicao = Derrota!')

    print('\n>>>>>> Prob. vitoria = %0.6f!' %(probVitoria))
    print('\n>>>>>> Prob. derrota = %0.6f!\n\n'  %(probDerrota))

# %% [markdown]
# ## Parte 4: Classificação de spam
# 
# Nesta parte do exercício, usaremos o Naive Bayes para classificar SMS spam.
# 
# Veja alguns exemplos de SMS legítimos:
#  * ```Is that seriously how you spell his name?```
#  * ```What you thinked about me. First time you saw me in class.```
#  * ```Ok lar i double check wif da hair dresser already he said wun cut v short. He said will cut until i look nice.```
#  
# Agora veja alguns exemplos de SMS spam:
#  * ```WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.```
#  * ```Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged.```
#  * ```Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.```

# %% [markdown]
# Antes de fazer qualquer tarefa de classificação com textos, é importante fazer um pré-processamento para obter melhor resultado na predição. Na função abaixo, os seguintes pré-processamentos são realizados:
# 
#  - deixar todas as palavras com letras minúsculas
#  - substituir os números pela palavra *number*
#  - substituir todas as URLS pela palavra *enderecoweb*
#  - substiuir todos os emails pela palavra *enderecoemail*
#  - substituir o símbolo de dólar pela palavra *dolar*
#  - substituit todos os caracteres não-alfanuméricos por um espaço em branco
#  
# Por fim, também é recomendado eliminar todas as palavras muito curtas. Vamos eliminar qualquer palavra de apenas 1 caracter. 

# %%
import re #regular expression

def preprocessing(text):
    
    # Lower case
    text = text.lower()
    
    # remove tags HTML
    regex = re.compile('<[^<>]+>')
    text = re.sub(regex, " ", text) 

    # normaliza os numeros 
    regex = re.compile('[0-9]+')
    text = re.sub(regex, "number", text)
    
    # normaliza as URLs
    regex = re.compile('(http|https)://[^\s]*')
    text = re.sub(regex, "enderecoweb", text)

    # normaliza emails
    regex = re.compile('[^\s]+@[^\s]+')
    text = re.sub(regex, "enderecoemail", text)
    
    #normaliza o símbolo de dólar
    regex = re.compile('[$]+')
    text = re.sub(regex, "dolar", text)
    
    # converte todos os caracteres não-alfanuméricos em espaço
    regex = re.compile('[^A-Za-z]') 
    text = re.sub(regex, " ", text)
    
    # substitui varios espaçamentos seguidos em um só
    text = ' '.join(text.split())
        
    return text

if __name__ == '__main__':
    smsContent = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.'
    print('Antes do preprocessamento: \n', smsContent)

    # chama a função de pré-processsamento para tratar o SMS
    smsContent = preprocessing(smsContent)

    print('\nDepois do preprocessamento: \n', smsContent)

# %% [markdown]
# Depois de fazer o pré-processamento, é necessário transformar o texto em um vetor de atributos com valores numéricos. Uma das formas de fazer isso é considerar que cada palavra da base de dados de treinamento é um atributo, cujo valor é o número de vezes que ela aparece em uma determinada mensagem.
# 
# Para facilitar, já existe um vocabulário no arquivo *vocab* que foi previamente extraído. Cada palavra desse vocabulário será considerado um atributo do problema de classificação de spam.
# 
# O código abaixo carrega o vocabulário.

# %%
if __name__ == '__main__':
    # Importa o vocabulario
    vocabulario = []
    with open(os.path.join(FILES_DIRECTORY, 'vocab.txt'), 'r') as f:
        for line in f:
            line = line.replace('\n','')

            vocabulario.append(line)

    # apresenta as primeiras palavras do vocabulário
    print('50 primeiras palavras do vocabulário:\n')
    print(vocabulario[0:50])

# %% [markdown]
# Agora, você deve completar a função abaixo para converter a mensagem em um vetor de atributos. O $i-ésimo$ atributo corresponderá à $i-ésima$ palavra do vocabulário e receberá como valor o número de vezes que ela aparecer na mensagem.
# 
# **Dica:**
# * Para contar o número de vezes que um valor aparece em uma lista, use: ``nomeLista.count(valor)``

# %%
import numpy as np 

def text2features(text, vocabulario):
    """
    Converte um texto para um vetor de atributos
    """
    
    #inicializa o vetor de atributos
    textVec = np.zeros( [1,len(vocabulario)], dtype=int )
    
    # faz a tokenização
    tokens = text.split() # separa as palavras com base nos espaços em branco
    
    # remove palavras muito curtas
    tokens = [w for w in tokens if len(w)>1]

    tam_voc = len(vocabulario)

    

    for i in range(tam_voc):
        textVec[0, i] = text.count(vocabulario[i])
        

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    # Complete esta função para retornar um vetor de atributos com valores numéricos
    # que represente o texto fornecido como entrada. 
    # O i-ésimo atributo corresponderá à i-ésima palavra do vocabulário e 
    # receberá como valor o numero de vezes que a palavra aparece na mensagem.
    #
    # Por exemplo, suponha que o texto de entrada contenha a palavra 'about'. Como essa palavra
    # é a 4 palavra do vocabulario, então a posição 3 do vetor de atributos deverá conter o valor.
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################

    return textVec


if __name__ == '__main__':
    # converte o texto para um vetor de features
    smsVec = text2features(smsContent, vocabulario)

    print('Vetor de features correspondente ao SMS:')
    print(smsVec[0:50])

# %% [markdown]
# ### Parte 4.1: Treinando o Naive Bayes
# 
# Nesta parte do exercício, nós iremos usar uma base de dados de treinamento que já foi pré-processada e convertida em vetores de atributos. O arquivo *spamTrain.txt* contém 4000 exemplos de emails spam e verídicos para treinamento. Por outro lado, o arquivo *spamTest.txt* contém 1000 exemplos de teste.
# 
# Primeiro, vamos carregar os arquivos.

# %%
import numpy as np

if __name__ == '__main__':
    # Importa o arquivo numpy
    dataset4_train = np.load(os.path.join(FILES_DIRECTORY, 'spamData.npz'))['train']
    dataset4_test = np.load(os.path.join(FILES_DIRECTORY, 'spamData.npz'))['test']

    # pega os valores das n-1 primeiras colunas e guarda em uma matrix X
    X4_train = dataset4_train[:, 0:-1]
    X4_test = dataset4_test[:, 0:-1]

    # pega os valores da última coluna e guarda em um vetor Y
    Y4_train = dataset4_train[:, -1] 
    Y4_test = dataset4_test[:, -1] 

    # imprimi as 5 primeiras linhas da matriz X
    display('X_train:', X4_train[0:5,:])
    display('X_test:', X4_test[0:5,:])

    # imprimi os 5 primeiros valores de Y
    print('Y_train:', Y4_train[0:5])
    print('Y_test:', Y4_test[0:5])

# %% [markdown]
# Vamos calcular qual a probabilidade de ocorrência de cada classe.

# %%
if __name__ == '__main__':
    # Probabilidade das Classes
    pSpam = sum(Y4_train==1)/len(Y4_train) 
    pHam = sum(Y4_train==0)/len(Y4_train)

    print('Probabilidade da classe ser 1 (spam): %1.2f%%' %(pSpam*100))
    print('Probabilidade da classe ser 0 (ham): %1.2f%%' %(pHam*100))

# %% [markdown]
# Agora, crie a função que irá calcular as probabilidades de ocorrência de cada atributo em cada classe. Use a correção de Laplace no cálculo da probabilidade de cada termo: 
# $$\hat{P}(w_i|c)=\frac{count(w_i|c)+1}{count(c)+|V|},$$
# onde $w_i$ é um termo do vocabulário, $count(c)$ é quantidade de termos nas amostras da classe $c$ e $|V|$ é o tamanho do vocabulário (número de atributos).

# %%
def calcularProbabilidades_Laplace(X, Y):
    """
    CALCULARPROBABILIDADES Computa a probabilidade de ocorrencia de cada 
    atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
    (qtde de atributos), um para cada classe.
    
    CALCULARPROBABILIDADES(X, Y) calcula a probabilidade de ocorrencia de cada atributo em cada classe. 
    Cada vetor de saida tem dimensao (n x 1), sendo n a quantidade de atributos por amostra.
    """
    
    #  inicializa os vetores de probabilidades
    pAtrSpam = np.zeros(X.shape[1])
    pAtrHam = np.zeros(X.shape[1])

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: Complete o codigo para encontrar a probabilidade de
    #                ocorrencia de um atributo para uma determinada classe.
    #                Ex.: para a classe 1 (vitoria), devera ser computada um
    #                vetor pAtrVitoria (n x 1) contendo n valores:
    #                P(Atributo1=1|Classe=1), ..., P(Atributo5=1|Classe=1), e o
    #                mesmo para a classe 0 (derrota):
    #                P(Atributo1=1|Classe=0), ..., P(Atributo5=1|Classe=0).
    #
    # 
    #basicamente a mesma formula, usamos o faltten junto ao sum para realizar a soma e usamos X.shape para conseguir o valor |V|
    countwc1 = (X[np.where(Y==1)]==1).sum(axis=0)
    countc1 = sum(X[Y==1].flatten())

    countwc0 = (X[np.where(Y==0)]==1).sum(axis=0)
    countc0 = sum(X[Y==0].flatten())
 

    pAtrSpam = (countwc1+1)/(countc1+X.shape[1])
    pAtrHam = (countwc0+1)/(countc0+X.shape[1])


    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################

    return pAtrSpam, pAtrHam


if __name__ == '__main__':
    pAtrSpam, pAtrHam = calcularProbabilidades_Laplace(X4_train,Y4_train)

    print('A probabilidade esperada para o primeiro atributo dada a classe spam = %.8f' %(0.00006613))
    print('\nEssa mesma probabilidade calculada no seu codigo foi = %.8f' %(pAtrSpam[0]))

# %% [markdown]
# Agora, vamos realizar a classificação das amostras com base nas probabilidades encontradas no passo anterior. A classificação é realizada verificando se a amostra em questão tem maior probabilidade de pertencer à classe 1 ou à classe 0. Conforme vimos no exercício anterior, para calcular a probabilidade de uma amostra pertencer a uma determinada classe, é necessário utilizar as probabilidades de ocorrências de atributos previamente computadas:
# 
# $$ P(y_j|\vec{x}) = \hat{P}(y_{j}) \prod_{x_i \in \vec{x}} \hat{P}(x_{i} | y_{j}) $$
# 
# Em classificação de textos, a probabilidade de ocorrência de cada termo geralmente é muito próxima de 0. Quando você multiplica essas probabilidades, o resultado final se aproxima ainda mais de 0, o que pode causar estouro de precisão numérica.
# 
# Um truque para evitar esse problema é substituir a equação acima por:
# 
# $$ P(y_j|\vec{x}) = \log\left(\hat{P}(y_{j})\right) + \sum_{x_i \in \vec{x}} \log\left(\hat{P}(x_{i} | y_{j})\right) $$
# 
# Você precisa completar a função abaixo para fazer a classificação de uma determinada mensagem em spam ou ham. Diferentemente da função feita no exercício anterior, o valor 0 não deve ser contabilizado no cálculo da probabilidade, pois ele significa que o termo não ocorreu na mensagem.
# 

# %%
def classificacao_texto(x,pSpam,pHam,pAtrSpam,pAtrHam):
    """
    Classifica se a entrada x pertence a classe 0 ou 1 usando
    as probabilidades extraidas da base de treinamento. Essa funcao 
    estima a predicao de x atraves da maior probabilidade da amostra  
    pertencer a classe 1 ou 0. Tambem retorna as probabilidades condicionais
    de vitoria e derrota, respectivamente.
    
    """

    #  inicializa a classe e as probabilidades condicionais
    classe = 0
    probSpam = 0
    probHam = 0

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: Complete o codigo para estimar a classificacao da amostra
    #                usando as probabilidades extraidas da base de treinamento.
    #                Voce precisara encontrar as probabilidades Bayesianas: 
    #                    . probVitoria -> p(classe=1|x)
    #                    . probDerrota -> p(classe=0|x) 
    #                Depois, você deve selecionar a maior delas. 
    # 

    #executando a formula um pouco diferente onde usamos log e somatorio
    probSpam = np.log(pSpam);
    probHam = np.log(pHam);
    for i in range(len(x)):
        if x[i] ==1:
            probSpam = probSpam + np.log(pAtrSpam[i])
            probHam = probHam + np.log(pAtrHam[i])

    if probSpam > probHam:
        classe = 1
    else:
        classe = 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    ########################################################################## 

    return classe, probSpam, probHam 


if __name__ == '__main__':
    resultados = np.zeros( X4_test.shape[0] )
    for i in range(X4_test.shape[0]):
        resultados[i], probSpam, probHam = classificacao_texto( X4_test[i,:],pSpam,pHam,pAtrSpam,pAtrHam )

    # calcular acuracia
    acuracia = np.sum(resultados==Y4_test)/len(Y4_test)

    print('\n\nAcuracia esperada para essa base = %.2f%%\n' %98.74)
    print('Acuracia obtida pelo seu classificador foi = %.2f%%\n' %( acuracia*100 ) )

# %% [markdown]
# Agora vamos testar o classificador que foi treinado em outro exemplo de SMS.

# %%
if __name__ == '__main__':
    smsContent = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.'

    print(smsContent) 

    # chama a função de pré-processsamento para tratar o email
    smsContent = preprocessing(smsContent)

    # converte o texto para um vetor de features
    smsVec = text2features(smsContent, vocabulario)

    # classifica o email
    classe, probSpam, probHam = classificacao_texto( smsVec[0,:],pSpam,pHam,pAtrSpam,pAtrHam )

    if classe==1:
        print('\n>>> Predicao = Spam!')       
    else:
        print('\n>>> Predicao = Ham!')

    print('\n>>>>>> Prob. spam = %0.18f!' %(probSpam))
    print('\n>>>>>> Prob. ham = %0.18f!\n\n'  %(probHam))


