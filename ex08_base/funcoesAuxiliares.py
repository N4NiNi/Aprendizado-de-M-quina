import numpy as np
import re

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

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    # Complete esta função para retornar um vetor de atributos com valores numéricos
    # que represente o texto fornecido como entrada. 
    # O i-ésimo atributo corresponderá à i-ésima palavra do vocabulário e 
    # receberá como valor o numero de vezes que a palavra aparece na mensagem.
    #
    # Por exemplo, suponha que o texto de entrada contenha a palavra 'about'. Como essa palavra
    # é a 4 palavra do vocabulario, então a posição 3 do vetor de atributos deverá conter o valor.
    
    for i in range( len(vocabulario) ):
        
        if vocabulario[i] in tokens:
            textVec[0,i] = tokens.count(vocabulario[i])
    
    ##########################################################################
    
    return textVec

