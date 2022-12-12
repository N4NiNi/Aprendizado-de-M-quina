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
# # <center>Exercício - Introdução ao Python</center>
# 
# Neste exercício, são apresentados alguns recursos básicos do Python que frequentemente são necessários nos projetos de Aprendizado de Máquina. 
# 
# Espera-se que, ao término deste notebook, você saiba realizar operações vetoriais com Numpy e manipular bases de dados com Pandas.

# %% [markdown]
# # <center>**Numpy**: vetores e matrizes</center>
# A biblioteca `numpy` permite a criação, manipulação e aplicação de operações matemáticas sobre vetores e matrizes. Ao longo da disciplina, recomenda-se a constante consulta da documentação da biblioteca, disponível [aqui](https://numpy.org/doc/stable/), que possui muitas funções interessantes e exemplos de uso.
# 
# Primeiro, você deve importar a biblioteca, antes de usar suas funções.

# %%
# -*- coding: utf-8 -*-
import numpy as np # importa a biblioteca usada para trabalhar com vetores e matrizes.

# %% [markdown]
# ## Criando vetores
# 
# A criação de vetores e matrizes com valores predefinidos pode ser feito através da função `np.array`.
# 
# Primeiro, vamos criar um vetor com os seguintes valores: $[1,2,3,4,5]$:

# %%
if __name__ == '__main__':
    # criando um vetor
    vet = np.array( [1,2,3,4,5] )

    # você pode imprimir usando a função print ou a função display
    print(vet)
    display(vet)

# %% [markdown]
# A biblioteca disponibiliza diversas funções para criação de vetores com elementos padronizados. Alguns exemplos são:
# 
# * `np.arange(M, N)`: cria um vetor com valor de $M$ a $(N-1)$;
# * `np.zeros(N)`: cria um vetor de 0s com tamanho $N$;
# * `np.ones(N)`: cria um vetor de 1s com tamanho $N$;
# * `np.repeat(E, N)`: cria um vetor com tamanho $N$ e conteúdo $E$;
# * `np.random.rand(N)`: cria um vetor com tamanho $N$ e conteúdo aleatório entre $(0,1)$
# 
# Vamos fazer alguns exemplos de vetores para testar as funções acima:

# %%
if __name__ == '__main__':
    # criando um vetor de 0 a 8
    vet_seq1 = np.arange(9)
    print('Vetor sequencial de 0 a 8:\t', vet_seq1)

    # criando um vetor de 5 a 10
    vet_seq2 = np.arange(5, 11)
    print('Vetor sequencial de 5 a 10:\t', vet_seq2)

    # criando um vetor de 0s de tamanho 5
    vet_zeros = np.zeros(5)
    print('Vetor de zeros:\t\t\t', vet_zeros)

    # criando um vetor de 1s de tamanho 3
    vet_ones = np.ones(3)
    print('Vetor de uns:\t\t\t', vet_ones)

    # criando um vetor de 8s de tamanho 4
    vet_repeat = np.repeat(8, 4)
    print('Vetor de elementos repetidos:\t', vet_repeat)

    # criando um vetor aleatório de tamanho 5
    vet_rand = np.random.rand(5)
    print('Vetor aleatorio:\t\t', vet_rand)

# %% [markdown]
# Todas estas funções ainda possuem diversos parâmetros para personalizarmos os elementos do vetor que será criado. Como exemplo, é possível usar `np.arange(M, N, S)` para criar vetores sequenciais de `M` a `(N-1)`variando de `S` em `S`.
# 
# Para aprender mais sobre as demais funções, recomenda-se o acesso a documentação da biblioteca.

# %%
if __name__ == '__main__':
    vet_seq1 = np.arange(5,21,2)
    print('Sequência de 5 até 20, variando de 2 em 2:', vet_seq1)

    vet_seq1 = np.arange(5,21,5)
    print('Sequência de 5 até 20, variando de 5 em 5:', vet_seq1)

    vet_seq1 = np.arange(5,21,4)
    print('Sequência de 5 até 20, variando de 4 em 4:', vet_seq1)

# %% [markdown]
# Vetores podem ser usados em laços de repetição. Como exemplo, vamos imprimir todos os valores entre 2 e 10 usando um laço de repetição `for`:

# %%
if __name__ == '__main__':
    for i in np.arange(2,11):
        print('%d' %i)

# %% [markdown]
# ## Criando matrizes
# 
# Para criar matrizes em numpy, basta criar um vetor com múltiplas dimensões, usando `np.array`.
# 
# _Obs.: embora a biblioteca possua uma classe específica para matrizes (`np.matrix`), recomenda-se para a disciplina o uso de vetores (`np.array`). Em numpy, vetores são N-dimensionais, sendo mais flexíveis que matrizes, que são estritamente bi-dimensionais. Como precisaremos dessa particularidade em exercícios futuros, é importante que você domine operações sobre vetores._
# 
# Vamos criar uma matriz com os valores: $
#   \begin{bmatrix}
#     1 & 2 & 3 & 4 & 5 \\
#     6 & 7 & 8 & 9 & 10
#   \end{bmatrix}$

# %%
if __name__ == '__main__':
    # cria uma matriz
    A = np.array( [[1,2,3,4,5],[6,7,8,9,10]] )

    print(A)

# %% [markdown]
# Assim como feito com vetores uni-dimensionais, nós também podemos criar uma matriz de valores zero e uma matriz de valores um:

# %%
if __name__ == '__main__':
    # cria um array de valores zeros com dimensão 2x10
    array_zeros = np.zeros( [2,10] )

    # cria um array de valores um com dimensão 2x10
    array_ones = np.ones( [2,10] )

    print('Vetor de zeros: ')
    print(array_zeros)

    print('\nVetor de valores um: ')
    print(array_ones)

# %% [markdown]
# Outras funções apresentadas, como `np.repeat` e `np.random.rand` também permitem a criação de matrizes. Para aprender mais sobre isso, consulte a documentação da biblioteca.

# %% [markdown]
# ## Selecionando elementos de vetores e matrizes
# 
# Há multiplas maneiras de selecionar elementos em vetores em python. Podemos selecionar um único elemento, utilizando o índice desse elemento, ou um subconjunto de elementos, operação chamada de _list slicing_.
# 
# Suponha que temos um vetor $[a,b,c,d,e,f,g,h,i]$. Nós podemos selecionar elementos e subconjuntos de elementos desse vetor conforme mostrado abaixo. 
# 
# **Atente-se**: em Python, o índice começa a contar em 0.

# %%
if __name__ == '__main__':
    # criando um vetor
    vet_abc = np.array(['a','b','c','d','e','f','g','h','i'])

    # acessando elementos do vetor
    print('O terceiro elemento de vet_abc:')
    print( vet_abc[3] )

    print('\nO último elemento de vet_abc:')
    print( vet_abc[-1] )

    print('\nTrês primeiros elementos de vet_abc: ')
    print( vet_abc[0:3] )

    print('\nTodos os valores após o quinto elementos de vet_abc: ')
    print( vet_abc[5:] )

    print('\nOs três ultimos valores de vet_abc: ')
    print( vet_abc[-3:] )

    print('\nOs valores de vet_abc entre o 5 elemento até o penúltimo elemento: ')
    print( vet_abc[4:-2], 'ou', vet_abc[4:7] )

# %% [markdown]
# O acesso a elementos de matrizes funciona de forma similar. Entretanto, podemos especificar o intervalo de seleção para cada dimensão de nosso vetor multi-dimensional. Sendo $V$ um vetor de 3 dimensões, indicamos os elementos e intervalos em cada uma das 3 dimensões usando a sintaxe $V[1,2,3]$. Caso desejemos capturar todos elementos de uma dimensão, podemos usar `:`.
# 
# Suponha que temos um array: $\begin{bmatrix}
#     1a & 1b & 1c & 1d & 1e & 1f & 1g & 1h & 1i \\
#     2a & 2b & 2c & 2d & 2e & 2f & 2g & 2h & 2i \\
#     3a & 3b & 3c & 3d & 3e & 3f & 3g & 3h & 3i \\
#     4a & 4b & 4c & 4d & 4e & 4f & 4g & 4h & 4i \\
#   \end{bmatrix}$.
# 
# Nós podemos selecionar vários subconjuntos de elementos dessa matriz, conforme mostrado abaixo.

# %%
if __name__ == '__main__':
    # criando uma matriz
    vet_abc = np.array( [['1a','1b','1c','1d','1e','1f','1g','1h','1i'],
                        ['2a','2b','2c','2d','2e','2f','2g','2h','2i'],
                        ['3a','3b','3c','3d','3e','3f','3g','3h','3i'],
                        ['4a','4b','4c','4d','4e','4f','4g', '4h','4i']])

    print('Matriz inteira: ')
    print( vet_abc )

    print('\nTodos os elementos da coluna 3: ')
    print( vet_abc[:,2] )

    print('\nTodos os elementos da linha 2: ')
    print( vet_abc[1,:] )

    print('\nTodos os elementos das 2 primeiras colunas: ')
    print( vet_abc[:,0:2] )

    print('\nTodos os elementos das 2 primeiras linhas: ')
    print( vet_abc[0:2,:] )

    print('\nApenas os elementos das 2 primeiras linhas e das 2 primeiras colunas: ')
    print( vet_abc[0:2,0:2] )

    print('\nApenas os elementos das 2 últimas linhas e das 4 últimas colunas: ')
    print( vet_abc[-2:,-4:] )

    print('\nApenas os elementos das linhas 2 até 4 e das colunas 4 até 6: ')
    print( vet_abc[1:3,3:6] )

# %% [markdown]
# ## Operações básicas com vetores e matrizes
# 
# Embora seja possível realizar operações entre vetores usando estruturas de repetição, isso é **extremamente ineficiente**! 
# 
# Utilizando funções do próprio numpy, podemos realizar essas operações de forma **vetorizada**. Essas funções são altamente otimizadas, executando muitas vezes operações em paralelo. Dessa forma, a execução do código se torna mais rápida.
# 
# Vamos aprender a realizar algumas operações básicas de forma vetorizada...

# %% [markdown]
# ### Soma e subtração
# 
# Podem ser usados os caracteres `+` e `-`, ou as funções `np.sum` e `np.subtract`.

# %%
if __name__ == '__main__':
    A = np.array( [1,2,3] )
    B = np.array( [4,5,6] )
    print('A:', A)
    print('B:', B)

    print('\nA+B: ', A+B)
    print('A-B: ', A-B)

# %% [markdown]
# Ambas as operações funcionam em matrizes da mesma maneira que em vetores:

# %%
if __name__ == '__main__':
    X1 = np.array( [[1,2,3],
                    [4,5,6]] )

    X2 = np.array( [[4,5,6],
                    [7,8,9]])

    print('X1: \n', X1);
    print('\nX2: \n', X2);

    print('\nX1 + X2:')
    print(X1+X2)

    print('\nX1 - X2:')
    print(X1-X2)

# %% [markdown]
# ### Produto entre duas matrizes
# 
# Através da função `np.dot` podemos realizar uma operação de **multiplicação matricial**:

# %%
if __name__ == '__main__':
    A = np.array( ([[1,2],[3,4],[5,6]]) )
    B = np.array( ([[1,2,3,4],[5,6,7,8]]) )

    print('A: ')
    display(A)

    print('B: ')
    display(B)

    print('A*B: ')
    print(np.dot(A,B)) 

# %% [markdown]
# Mas **atenção**, diferente da adição e subtração, não podemos usar `*` para multiplicar duas matrizes. O asterisco é utilizado para **multiplicação ponto-a-ponto**, funcionando assim apenas para matrizes de tamanhos idênticos:

# %%
if __name__ == '__main__':
    A = np.array( ([[1,2],[3,4],[5,6]]) )
    B = np.array( ([[0,1],[2,3],[4,5]]) )

    print('A: ')
    display(A)

    print('B: ')
    display(B)

    print('A*B: ')
    print(A*B) 

# %% [markdown]
# ### Média dos valores do vetor
# 
# Para calcular a média entre os elementos de um vetor, podemos usar a função `np.mean`:

# %%
if __name__ == '__main__':
    A = np.array( [2, 3, 5, 7, 11, 13, 17] )

    print('A: ', A)

    print('Média: ', np.mean(A))

# %% [markdown]
# Podemos também usar `np.mean` para calcular a média entre os elementos de uma **matriz**. Adicionalmente, podemos usar o parâmetro `axis` para calcular a média por linha ou por coluna.
# 
# Muitas funções do numpy, incluindo `np.sum`, `np.subtract` e `np.mean`, possuem a opção de realizar a operação a nível de linha ou coluna por meio de `axis`
# * `axis=0`: realiza a operação a nível de **coluna**
# * `axis=1`: realiza a operação a nível de **linha**

# %%
if __name__ == '__main__':
    print('B: ')
    print(B)

    # média das colunas de B
    media_coluna = np.mean(B, axis = 0)
    print('\nMédia das colunas de B: ')
    print(media_coluna)

    # média das linhas de B
    media_linha = np.mean(B, axis = 1)
    print('\nMédia das linhas de B: ')
    print(media_linha)

    # média de todos os valores de B 
    media_geral = np.mean(B)
    print('\nMédia de todos os valores de B: ')
    print(media_geral)

# %% [markdown]
# ### Desvio padrão
# 
# Para calcular o desvio padrão entre os elementos de um vetor podemos usar `np.std`. Assim como nos últimos exemplos, a função também possui o parâmetro `axis`. 
# 
# Adicionalmente, a função possui o parâmetro `ddof`, que representa os graus de liberdade (_delta degrees of freedom_). Usando `ddof=0`, é calculado o desvio padrão populacional. Para calcular o desvio padrão amostral, considerado estatisticamente mais apropriado para as aplicações da disciplina, temos que usar `ddof=1`.

# %%
if __name__ == '__main__':
    print('B: ')
    print(B)

    # desvio padrão das linhas de B
    std1 = np.std(B, axis = 1, ddof=1)
    print('\nDesvio padrão das linhas de B: ')
    display(std1)

    # desvio padrão das colunas de B
    std2 = np.std(B, axis = 0, ddof=1)
    print('\nDesvio padrão das colunas de B: ')
    display(std2)

    # desvio padrão de todos os valores de B 
    std3 = np.std(B, ddof=1)
    print('\nDesvio padrão de todos os valores de B: ')
    display(std3)

# %% [markdown]
# ### Outras funções para vetores e matrizes
# 
# A biblioteca possui ainda diversas outras funções vetorizadas. Alguns exemplos que serão bastante usados na disciplina são:
# 
# * `np.min()` &rarr; valor mínimo
# * `np.max()` &rarr; valor máximo
# * `np.sort(A)` &rarr; retorna o vetor $A$ ordenado
# * `np.argsort(A)` &rarr; retorna os índices do vetor $A$ ordenado
# * `np.var()` &rarr; variância
# * `np.shape()` &rarr; dimensões da matriz
# * `np.transpose()` &rarr; transposta da matriz
# * `np.concatenate(arrays, axis=numero_do_eixo)` &rarr; concatena vetores ou matrizes
# * `vstack(A,B)` &rarr; empilha verticalmente
# * `hstack(A,B)` &rarr; empilha horizontalmente
# * `np.where(A>n)` &rarr; elementos em $A$ maiores que $n$

# %% [markdown]
# ## **Exercícios - Numpy**
# 
# A seguir, são propostos diversos exercícios para testar o que você aprendeu sobre numpy. Em cada um deles, você deve preencher a função **apenas dentro do espaço indicado**.
# 
# Atente-se para fazer a sua implementação o mais genérica possível antes de enviar no _online judge_. O corretor avaliará com valores diferentes dos passados neste notebook.

# %% [markdown]
# **Ex. 1**. Crie duas matrizes com os nomes $ExA$ e $ExB$ preenchidas com os valores passados como parâmetro.

# %%
def criaMatrizes( val_A, val_B ):
    """
    Gera duas matrizes com conteudo predefinido
    """
    
    ExA = np.array([[0, 0]])
    ExB = np.array([[0, 0]])
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    ExA = np.array(val_A)
    ExB = np.array(val_B)
    
    ##########################################################################

    return ExA, ExB


if __name__ == '__main__':
    val_A = [[12,9,4,1],[11,5,8,1],[1,2,3,1]]
    val_B = [[1,5],[1,7],[2,9],[1,1]]

    ExA, ExB = criaMatrizes(val_A, val_B)

    print('ExA:')
    print(ExA)

    print('\nExB:')
    print(ExB)

# %% [markdown]
# **Ex. 2**. Crie uma matriz $ExC$ usando a seguinte operação: $ExC = ExA \cdot ExB$.

# %%
def multiplicaMatrizes( ExA, ExB ):
    """
    Gera uma matriz atraves da multilicacao de outras duas
    """
    
    ExC = np.array([[0, 0]])
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    ExC = np.dot(ExA,ExB)
    
    
    
    
    
    ##########################################################################

    return ExC


if __name__ == '__main__':
    ExC = multiplicaMatrizes(ExA, ExB)

    print('ExC:')
    print(ExC)

# %% [markdown]
# **Ex. 3**. Crie uma função que retorne a média e o desvio padrão das linhas de uma matriz e a média e desvio padrão das colunas de uma matriz.

# %%
def mediaStdMatriz( M ):
    """
    Calcula a media e desvio padrao das linhas e colunas da matriz M
    """

    media_linhas = media_colunas = std_linhas = std_colunas = 0    
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    media_linhas = np.mean(M, axis = 1)
    std_linhas = np.std(M, axis=1,ddof=1)

    media_colunas = np.mean(M, axis = 0)
    std_colunas = np.std(M, axis=0,ddof=1)  
  
    ##########################################################################

    return media_linhas, media_colunas, std_linhas, std_colunas


if __name__ == '__main__':
    media_linhas, media_colunas, std_linhas, std_colunas = mediaStdMatriz(ExC)

    print('Medias linhas: ', media_linhas)
    print('Medias colunas: ', media_colunas)

    print('Desvio padrão linhas: ', std_linhas)
    print('Desvio padrão colunas: ', std_colunas)

# %% [markdown]
# **Ex. 4**. Crie uma função que gera uma matriz $ExD$ com os valores das duas últimas colunas da matriz passada como parâmetro. Depois, calcule a média geral dos valores de $ExD$.

# %%
def duasUltimasColunasMedia( M ):
    """
    Gera uma matriz com os valores das duas ultimas colunas de M e calcula a média geral
    """
    
    ExD = np.array([[0, 0]])
    media_D = 0
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    ExD = M[:,-2:]
    media_D = np.mean(ExD)
    
    ##########################################################################

    return ExD, media_D


if __name__ == '__main__':
    ExD, media_D = duasUltimasColunasMedia(ExA)

    print('ExD:')
    print(ExD)

    print('\nMédia')
    print(media_D)

# %% [markdown]
# **Ex. 5**. Crie uma função que gera matriz $ExE$ com os valores das linhas de índice 1 e 2 e das colunas de índice 1 e 2 de uma matriz passada como parâmetro (_lembre-se que o índice começa em 0_).

# %%
def matrizLinhasColunas( M ):
    """
    Gera uma matriz com os valores das linhas e colunas de M com indice 1 e 2
    """
    
    ExE = np.array([[0, 0]])    
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    ExE = M[1:3,1:3]
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################

    return ExE


if __name__ == '__main__':
    ExE = matrizLinhasColunas(ExA)

    print('ExE:')
    print(ExE)

# %% [markdown]
# **Ex. 6**. Crie uma função que gera uma matriz $ExF$ com $M$ linhas e $N$ colunas e todos os valores iguais a 0.

# %%
def matrizZeros( M, N ):
    """
    Gera uma matriz com M linhas e N colunas, preenchendo seu conteudo com 0
    """
    
    ExF = np.array([[0, 0]])    
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    ExF = np.zeros([M,N])

    ##########################################################################

    return ExF


if __name__ == '__main__':
    ExF = matrizZeros(5, 2)

    print('ExF:')
    print(ExF)

# %% [markdown]
# **Ex. 7**. Crie uma função que gera um vetor $ExG$ com $M$ elementos e todos os valores iguais a $V$.

# %%
def vetorVs( M, V ):
    """
    Gera um vetor com M elementos, preenchendo seu conteudo com V
    """
    
    ExG = np.array([0, 0])    
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    ExG = np.repeat(V,M)
    
    
    ##########################################################################

    return ExG


if __name__ == '__main__':
    ExG = vetorVs(4, 3)

    print('ExG:')
    print(ExG)

# %% [markdown]
# **Ex. 8**. Crie uma função que retorne o fatorial de um número. Teste a função que você criou, calculando o fatorial de 8. 
# 
# Fórmula do fatorial: $n! = (1 * 2 * 3 * ... * (n-2) * (n-1) * n)$
# 
# **Obs: é obrigatório usar um laço `for`.**
# 
# Exemplos: 
# * $1! = 1$
# * $2! = 1*2$
# * $3! = 1*2*3$
# * $4! = 1*2*3*4$
# * $5! = 1*2*3*4*5$

# %%
def fatorial(n):
    """
    Calcula o fatorial de n (n!)
    """
    
    fat = 0
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    fat=1
    for i in range(n):
        fat=fat*(i+1)

    ##########################################################################
    
    return fat
    

if __name__ == '__main__':
    fat = fatorial(8)

    print('8! = ', fat)

# %% [markdown]
# # <center>**Pandas**: conjuntos de dados</center>
# 
# A biblioteca `pandas` permite a criação e manipulação de conjuntos de dados estruturados, tratados como uma matriz. Ao longo da disciplina, recomenda-se a constante consulta da documentação da biblioteca, disponível [aqui](https://pandas.pydata.org/), que possui muitas funções interessantes e exemplos de uso.
# 
# Primeiro, você deve importar a biblioteca, antes de usar suas funções.

# %%
# -*- coding: utf-8 -*-
import pandas as pd # importa a biblioteca usada para trabalhar com conjuntos de dados

# %% [markdown]
# ### Criando DataFrames
# 
# Enquanto numpy utiliza vetores, o principal objeto utilizado pelo pandas é chamado DataFrame. Um DataFrame consiste em um vetor bi-dimensional (sendo um _np.array_ por baixo dos panos), com informações adicionais, como nomes de coluna e índice.
# 
# Para criar um DataFrame com valores predefinidos, podemos usar a função `pd.DataFrame`, passando um _np.array_ como parâmetro. Vamos criar um DataFrame, definindo nomes para suas colunas e índices (linhas):

# %%
if __name__ == '__main__':
    # cria uma matriz
    vec_A = np.array( [[1,2,3,4,5],[6,7,8,9,10]] )
    print('Matriz:')
    print(vec_A)

    # nomes das linhas e colunas
    index_names = ['Ind1', 'Ind2']
    column_names = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']

    # gera o dataframe
    df_A = pd.DataFrame(vec_A, index=index_names, columns=column_names)
    print('\nDataFrame:')
    display(df_A)

# %% [markdown]
# É possível obter o conteúdo de um _pd.DataFrame_ em formato _np.array_ através do atributo `values`:

# %%
if __name__ == '__main__':
    print('Conteúdo do DataFrame como array:')
    print(df_A.values)

# %% [markdown]
# Na Ciência de Dados e em aplicações de Aprendizado de Máquina, é comum usarmos DataFrames para manipular os conjuntos de dados. Normalmente, estes conjuntos são salvos em arquivos de diferentes formatos, e através do pandas, podemos gerar um DataFrame direto da leitura de um arquivo. Alguns exemplos de funções para essa finalidade são:
# 
# * `pd.read_csv`: gera um DataFrame de um arquivo CSV
# * `pd.read_excel`: gera um DataFrame de um arquivo XLS
# * `pd.read_json`: gera um DataFrame de um arquivo JSON
# * `pd.read_sql`: gera um DataFrame com base em uma query SQL em um banco de dados
# 
# entre outras...

# %% [markdown]
# ### Acessando linhas e colunas pelo nome
# 
# Para acessar uma **coluna** específica do DataFrame pelo nome, usamos `df['col']`, onde `df` corresponde ao DataFrame e `'col'` ao nome da coluna.

# %%
if __name__ == '__main__':
    # acessa a terceira coluna
    print('Terceira coluna do DataFrame: ')
    display(df_A['Col3'])

# %% [markdown]
# Para acessar uma **linha**, usamos o atributo `loc`, da seguinte forma: `df.loc['index']`, onde `'index'` corresponde ao nome da linha. É também possível acessar a coluna por meio de `loc`, seguindo uma lógica semelhante ao acesso em vetores multi-dimensionais em numpy: `df.loc[:, 'col']`.

# %%
if __name__ == '__main__':
    # acessa a primeira linha
    print('Primeira linha do DataFrame: ')
    display(df_A.loc['Ind1'])

    # acessa a segunda coluna usando loc
    print('\nSegunda coluna do DataFrame usando loc: ')
    display(df_A.loc[:, 'Col2'])

# %% [markdown]
# Por fim, é possível acessar uma posição específica usando `loc` e passando ambos os nomes da coluna e da linha.

# %%
if __name__ == '__main__':
    # acessa o elemento na segunda linha e quarta coluna
    print('Elemento na segunda linha e quarta coluna: ')
    print(df_A.loc['Ind2', 'Col4'])

# %% [markdown]
# ### Acessando linhas e colunas pela posição
# 
# Uma outra maneira de acessar o conteúdo de um DataFrame é pela posição da linha e coluna. Isso é muito útil para quando você não sabe o nome da linha/coluna, ou sabe que este nome pode mudar ao longo da execução.
# 
# Para acessar um elemento pela posição, usamos `iloc`, de forma muito semelhante ao `loc`:
# 
# _Obs.: lembre-se que o índice em Python começa em 0_

# %%
if __name__ == '__main__':
    # acessa a terceira coluna
    print('\nTerceira coluna do DataFrame: ')
    display(df_A.iloc[:, 2])

    # acessa a primeira linha
    print('\nPrimeira linha do DataFrame: ')
    display(df_A.iloc[0])

    # acessa o elemento na segunda linha e quarta coluna
    print('\nElemento na segunda linha e quarta coluna: ')
    print(df_A.iloc[1, 3])

# %% [markdown]
# ### Operações matriciais dentro de um DataFrame
# 
# Assim como ocorre em numpy, a biblioteca pandas fornece diversas funções para realizar operações de forma vetorizada dentro do conteúdo do DataFrame. Nelas, também utilizamos o parâmetro `axis` para definir se as operações são realizadas a nível de linha ou coluna. Adicionalmente, algumas das funções que vimos em numpy podem ser usadas recebendo um DataFrame como parâmetro.
# 
# Vamos realizar alguns exemplos de operações sobre seus elementos:

# %%
if __name__ == '__main__':
    # soma as colunas do DataFrame
    print('Soma das colunas:')
    display(df_A.sum(axis=0))

    # soma todos os elementos do DataFrame
    print('\nSoma geral do DataFrame:')
    print(df_A.sum().sum())

    # calcula a média por linha
    print('\nMédia das linhas:')
    display(df_A.mean(axis=1))

    # calcula o desvio padrão por colunas
    print('\nDesvio padrão das colunas:')
    display(df_A.std(axis=0, ddof=1))

    # calcula a mediana das linhas
    print('\nMediana das linhas:')
    display(df_A.median(axis=1))

# %% [markdown]
# ### Multiplicação de DataFrames
# 
# Assim como feito com vetores, podemos usar o método `np.dot` para realizar multiplicação matricial entre dois DataFrames. 
# 
# _Obs.: embora a biblioteca possua o método `pd.DataFrame.dot`, ele requer que as colunas e índices tenham o mesmo nome, o que não é o caso de DataFrames com dimensões diferentes entre si. Por esse motivo, é preferível usar `pd.DataFrame.values` com `np.dot`._

# %%
if __name__ == '__main__':
    # cria uma matriz
    vec_B = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]])
    print('Matriz:')
    print(vec_B)

    # nomes das linhas e colunas
    index_names = ['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5']
    column_names = ['Col1', 'Col2', 'Col3']

    # gera o dataframe
    df_B = pd.DataFrame(vec_B, index=index_names, columns=column_names)

    # Exibe os dataframes
    print('\nDataFrame A:')
    display(df_A)
    print('\nDataFrame B:')
    display(df_B)

    # multiplicacao matricial
    print('\nA * B')
    display(pd.DataFrame(np.dot(df_A.values, df_B.values)))

# %% [markdown]
# ### Tratamento e análise de dados com pandas
# 
# A biblioteca fornece muitas funções comumente utilizadas na etapa de exploração e tratamento dos dados. Não veremos como elas funcionam neste notebook porque serão explicadas em notebooks futuros da aula, além de estarem descritas na documentação da biblioteca. Alguns exemplos são:
# 
# * `fillna`: preenche posições vazias do DataFrame com valores específicos, como uma valor padrão ou a média;
# * `interpolate`: preenche posições vazias com valores interpolados
# * `drop_duplicates`: remove linhas que contenham valores duplicados
# * `sort_values`: ordena o DataFrame por colunas específicas
# * `value_counts`: conta a quantidade de ocorrência de valores únicos no DataFrame
# * `pivot_table`: gera um novo DataFrame usando o conteúdo do DataFrame original como índices e colunas
# * `groupby`: agrupa o DataFrame considerando o conteúdo de alguma coluna e aplicando funções de agregação nas demais

# %% [markdown]
# ## **Exercícios - Pandas**
# 
# A seguir, são propostos dois exercícios para testar o que você aprendeu sobre pandas. Em cada um deles, você deve preencher a função **apenas dentro do espaço indicado**.
# 
# Atente-se para fazer a sua implementação o mais genérica possível antes de enviar no online judge. O corretor avaliará com valores diferentes dos passados neste notebook.

# %% [markdown]
# **Ex. 9**. Crie uma função que retorna o elemento de uma posição específica de um DataFrame `df` com base nos nomes da coluna e linha.

# %%
def pegaElementoPeloNome( df, C_name, I_name ):
    """
    Recupera o elemento do DataFrame df com base no nome da coluna e da linha, denotados por C_name e I_name
    """
    
    elem = 0
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    elem = df.loc[I_name,C_name]
    
    ##########################################################################

    return elem


if __name__ == '__main__':
    df = pd.DataFrame(
        [[10, 33, 42], [54, 11, 97], [50, 28, 87], [44, 19, 83]],
        columns=['Num1', 'Num2', 'Num3'],
        index=['AM', 'IA', 'NLP', 'RNA']
    )
    display(df)

    elem = pegaElementoPeloNome(df, 'Num1', 'IA')

    print('Elemento da coluna Num1 e da linha IA: ', elem)

# %% [markdown]
# **Ex. 10**. Crie uma função que retorna o elemento de uma posição específica de um DataFrame `df` com base nos índices da coluna e linha.

# %%
def pegaElementoPelaPosicao( df, C_pos, I_pos ):
    """
    Recupera o elemento do DataFrame df com base no nome da coluna e da linha, denotados por C_name e I_name
    """
    
    elem = 0
    
    ########################## COMPLETE O CÓDIGO AQUI  #######################
    
    elem = df.iloc[I_pos,C_pos]
    
 
    
    ##########################################################################

    return elem


if __name__ == '__main__':
    df = pd.DataFrame(
        [[10, 33, 42], [54, 11, 97], [50, 28, 87], [44, 19, 83]],
        columns=['Num1', 'Num2', 'Num3'],
        index=['AM', 'IA', 'NLP', 'RNA']
    )
    display(df)

    elem = pegaElementoPelaPosicao(df, 1, 0)

    print('Elemento da coluna 1 e da linha 0: ', elem)


