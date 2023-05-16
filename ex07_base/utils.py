import numpy as np

def inicializaPesos(fan_out, fan_in):
    '''
    Inicializa os pesos para uma camada com fan_in
    (numero de conexoes de entrada) e fan_out (numero de conexoes de saida)
    usando uma estrategia fixa, ajudando a testar seu codigo
    sem aleatoriedade embutida.
    
    Observe que W sera definido como uma matriz do tamanho (fan_out, 1 + fan_in)
    por conta que a primeira coluna eh separada para o "bias"
    '''

    # Define W como matriz de zeros
    W = np.zeros([fan_out, 1 + fan_in])

    # Inicializa W usando a funcao "sin", para garantir sempre os mesmos valores
    W = np.reshape(np.sin(np.arange(1,np.prod(W.shape)+1)), W.shape) / 10

    # =========================================================================

    return W

def gradienteNumerico(J, theta, e=1e-4):
    """
    Calcula o gradiente usando "diferencas finitas" e 
    da um resultado estimado do gradiente.

    Parametros
    ----------
    J : funcao de custo que sera usada para estimar o gradiente numerico

    theta : os pesos da rede neural. O gradiente numerico e calculado
            com base nesses pesos

    e : o valor de epsilon que é usado para calcular as "diferencas finitas".



    Notas
    -----
    O codigo a seguir implementa a checagem do gradiente numerico
    e retorna o gradiente numerico. O valor numgrad[i] se refere
    a uma aproximacao numerica da derivada parcial de J com relacao
    ao i-esimo argumento de entrada. Ou seja, numgrad[i] se
    refere a derivada parcial aproximada de J com relacao a theta[i].
    """

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    
    eps = 1e-4
    
    for p in range(len(theta)):
        perturb[p] = eps
        
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        
        # Calcula o gradiente numerico
        numgrad[p] = (loss2 - loss1) / (2*eps)
        perturb[p] = 0
        
    return numgrad


def verificaGradiente(nnCostFunction, vLambda=None):
    """
    Cria uma pequena rede neural para verificar 
    os gradientes de backpropagation. Serao exibidos o gradiente produzido
    pelo seu codigo de backpropagation e o gradiente numerico obtido 
    na funcao gradienteNumerico. Ambos gradientes devem ser bem proximos.
    """
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = inicializaPesos(hidden_layer_size,input_layer_size)
    Theta2 = inicializaPesos(num_labels,hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = inicializaPesos(m, input_layer_size - 1)

    y = np.arange(1, 1+m) % num_labels
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([np.ravel(Theta1), np.ravel(Theta2)])
    

    if vLambda is None:
        # short hand for cost function
        costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y)
    else:
        # short hand for cost function
        costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, vLambda)
        
    cost, grad = costFunc(nn_params)
    numgrad = gradienteNumerico(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(np.stack([numgrad, grad], axis=1))
    print('As duas colunas acima deve ser bem semelhantes.')
    print('(Esquerda - Gradiente numerico, Direita - Seu gradiente)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('Se sua implementacao de backpropagation esta correta, \n'
          'a diferenca relativa devera ser pequena (menor que 1e-9). \n'
          '\nDiferenca relativa: %g\n' % diff)


#if __name__ == "__main__":
#    
#    print('\nChecando Backpropagation... \n')
#    
#    # Verifica o gradiente usando a funcao verificaGradiente
#    verificaGradiente()
#    
#    print('\nVerificando backpropagation (c/ regularizacao) ... \n')
#    
#    # Verifica gradiente usando a funcao verificaGradiente
#    #vLambda = 3
#    #verificaGradiente(funcaoCusto_backp_reg)
    
    