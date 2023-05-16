import numpy as np
import scipy
import scipy.optimize 

def atributosPolinomiais(X1,X2):
    """
    Gera atributos polinomiais a partir dos atriburos
    originais da base

    ATRIBUTOSPOLINOMIAIS(X1, X2) mapeia os dois atributos de entrada
    para atributos quadraticos
 
    Retorna um novo vetor de mais atributos:
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
 
    As entradas X1, X2 devem ser do mesmo tamanho
    """
    
    grau=6
    
    # se se X1 é um array. Se não for, converte para array
    if not isinstance(X1,  np.ndarray):
        X1 = np.array( [[X1]] )
        X2 = np.array( [[X2]] )
        
    out = np.ones( len(X1) )
    
    for i in range( 1,grau+1 ):
        for j in range( 0,i+1 ):
            out_temp = ( X1**(i-j) ) * (X2**j)
            
            out = np.column_stack( (out,out_temp) ) # Adicionar uma coluna de 1s em x

    return out

def funcaoCustoReg(theta, X, Y, lambda_reg):
    """
    Calcula o custo da regressao logística
    
       J = COMPUTARCUSTO(X, y, theta) calcula o custo de usar theta como 
       parametro da regressao logistica para ajustar os dados de X e y    
    """
    
    # Initializa algumas variaveis uteis
    m = len(Y) #numero de exemplos de treinamento

    # Voce precisa retornar a seguinte variavel corretamente
    J = 0
    grad = np.zeros( len(theta) )
    
    # ====================== ESCREVA O SEU CODIGO AQUI ======================
    # Instrucoes: Calcule o custo de uma escolha particular de theta.
    #             Voce precisa armazenar o valor do custo em J.
    #             Calcule as derivadas parciais e encontre o valor do gradiente
    #             para o custo com relacao ao parametro theta
    # Obs: grad deve ter a mesma dimensao de theta

    # parâmetro de tolerância para a função sigmoide. 1-htheta não pode ser 
    # menor que eps para evitar erro de precisão numérica
    eps = 1e-15
    
    reg = lambda_reg/(2*m) * np.sum(theta[1:] ** 2)
    
    h = sigmoid( np.dot(X,theta) )
    
    J = np.dot( -Y,np.log(h) ) - np.dot( (1-Y),np.log(1-h+eps) )
    J = (1/m)*J + reg
    
    grad = np.dot( (h-Y),X ) 
    grad = (1/m) * grad
    
    # usa a regularização no gradiente
    grad[1:] = grad[1:] + (lambda_reg/m) * theta[1:]
                  

    #=========================================================================
    
    return J, grad

def sigmoid(z):
    """
    Calcula a funcao sigmoidal  
    """
    
    # Voce precisa retornar a variável g corretamente
    #
    # se z for um valor inteiro, inicialize g com 0
    if isinstance(z, int):
        g = 0
    
    # se z não é um inteiro, significa que é um array e inicia com a dimensão do array
    else:
        g = np.zeros( z.shape );

    # ====================== ESCREVA O SEU CODIGO AQUI ======================
    # Instrucoes: Calcule a sigmoid de cada valor de z (z pode ser uma matriz,
    #              vetor ou escalar).

    g = 1/(1+np.exp(-z))

    #=========================================================================
    
    return g

def predicao(X, theta):
    """
    Prediz se a entrada pertence a classe 0 ou 1 usando o parametro
    theta obtido pela regressao logistica
    
    p = PREDICAO(theta, X) calcula a predicao de X usando um 
    limiar igual a 0.5 (ex. se sigmoid(theta'*x) >= 0.5, classe = 1)
    """   
    
    m = X.shape[0] # Numero de examplos de treinamento
    
    # Voce precisa retornar a seguinte variavel corretamente
    p = np.zeros(m, dtype=int)
    
    
    # ====================== ESCREVA O SEU CODIGO AQUI ======================
    # Instrucoes: Complete o codigo a seguir para fazer predicoes usando
    # os paramentros ajustados pela regressao logistica. 
    # p devera ser um vetor composto por 0's e 1's
    
    h = sigmoid( np.dot(X,theta) )
    
    for i in range(m):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    # =========================================================================
    
    return p
    
def treinamento(X, Y, lambda_reg, iteracoes):
    
    # se for vazio, retorna None 
    if len(Y)==0:
        return None
    
    m, n = X.shape # m = qtde de objetos e n = qtde de atributos por objeto
    
    theta = np.zeros(n) # Inicializa parâmetros que serao ajustados
    
    # minimiza a funcao de custo
    result = scipy.optimize.minimize(fun=funcaoCustoReg, x0=theta, args=(X, Y, lambda_reg),  
                method='BFGS', jac=True, options={'maxiter': iteracoes, 'disp':False})

    # coleta os thetas retornados pela função de minimização
    theta = result.x
    
    return theta
    
