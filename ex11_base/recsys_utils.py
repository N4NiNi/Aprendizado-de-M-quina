import csv
from datetime import datetime
import os
import pandas as pd

# ========== CONSTANTES ==========

# Pastas
CLEAN_DATASET_FOLDER = 'dataset'

# Arquivos
RAW_GENRE_FILE = 'u.genre'
RAW_ITEM_FILE = 'u.item'
RAW_USER_FILE = 'u.user'
RAW_INTERACTIONS_FILE = 'u.data'
CLEAN_ITEM_FILE = 'items.csv'
CLEAN_USER_FILE = 'users.csv'
CLEAN_INTERACTIONS_FILE = 'interactions.csv'

# Dados dos csv dos arquivos finais
CLEAN_FILE_STYLE = {
    'quotechar': '"',
    'quoting': csv.QUOTE_ALL,
    'sep': ';',
    'header': True,
    'index': False,
    'encoding': 'latin-1'
}

# ========== FUNCOES ==========

# Limpa e prepara a base do Movielens
def format_movielens_dataset(raw_dataset_folder='ml-100k', sampling_rate=1.0):
    print('Iniciando tratamento da base Movielens...')

    # Verifica se a pasta com o dataset original existe
    if not os.path.exists(raw_dataset_folder):        
        raise Exception('A pasta com os dados originais do Movielens ({}) não foi encontrada'.format(raw_dataset_folder))
    
    # Formata os itens
    print('\tTratando os itens...')
    # Carrega a base de generos dos filmes
    genres = pd.read_csv(os.path.join(raw_dataset_folder, RAW_GENRE_FILE), sep='|', encoding='latin-1', header=None)
    genres.columns = ['nome', 'id']
    # Remove o genero "unknown"
    genres = genres[genres['nome']!='unknown']
    # Carrega a base de filmes
    items = pd.read_csv(os.path.join(raw_dataset_folder, RAW_ITEM_FILE), sep='|', encoding='latin-1', header=None)
    # Remove colunas nao utilizadas
    items = items.drop(columns=[2, 3, 4, 5]) # A 5 diz respeito ao genero "unknown"
    # Arruma o nome das colunas e adiciona as colunas de generos
    items.columns = ['id_item', 'titulo-ano'] + list(genres['nome'])
    # Trata um caso especifico que nao esta seguindo o padrao titulo-ano
    items.loc[1411, 'titulo-ano'] = items.loc[1411, 'titulo-ano'][:-3]
    # Separa o titulo e o ano
    items['titulo'] = items['titulo-ano'].apply(lambda x: x.strip()[:-6])
    items['ano'] = items['titulo-ano'].apply(lambda x: x.strip()[-5:-1])
    # Agrupa os generos numa unica coluna
    items['genero'] = items[genres['nome']].apply(lambda x: '/'.join(x[x==1].index), axis='columns')        
    # Filtra apenas as colunas relevantes
    items = items[['id_item', 'titulo', 'ano', 'genero']]

    # Formata os usuarios
    print('\tTratando os usuarios...')    
    users = pd.read_csv(os.path.join(raw_dataset_folder, RAW_USER_FILE), sep='|', encoding='latin-1', header=None)    
    users.columns = ['id_usuario', 'idade', 'genero', 'ocupacao', 'zip']    

    # Formata as interacoes
    print('\tTratando as interacoes...')    
    interactions = pd.read_csv(os.path.join(raw_dataset_folder, RAW_INTERACTIONS_FILE), sep='\t', encoding='latin-1', header=None)    
    interactions.columns = ['id_usuario', 'id_item', 'nota', 'timestamp']
    # Transforma o timestamp em data-hora
    interactions['data-hora'] = interactions['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    # Filtra apenas as colunas relevantes
    interactions = interactions[['id_usuario', 'id_item', 'nota', 'data-hora']]

    # Remove o filme 267 que esta vazio    
    items = items[items['id_item']!=267]
    interactions = interactions[interactions['id_item']!=267]
    
    # Seleciona uma amostragem das interacoes
    interactions = interactions.sample(int(len(interactions)*sampling_rate), random_state=7)
    
    # Limpa usuarios e filmes removidos do dataframe de interacoes
    items = items[items['id_item'].isin(interactions['id_item'])]
    users = users[users['id_usuario'].isin(interactions['id_usuario'])]    

    # Salva os arquivos formatados
    print('\tSalvando os arquivos tratados...')
    os.makedirs(CLEAN_DATASET_FOLDER, exist_ok=True)
    items.to_csv(os.path.join(CLEAN_DATASET_FOLDER, CLEAN_ITEM_FILE), **CLEAN_FILE_STYLE)
    users.to_csv(os.path.join(CLEAN_DATASET_FOLDER, CLEAN_USER_FILE), **CLEAN_FILE_STYLE)
    interactions.to_csv(os.path.join(CLEAN_DATASET_FOLDER, CLEAN_INTERACTIONS_FILE), **CLEAN_FILE_STYLE)
    print('Tudo OK!')