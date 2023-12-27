#import das bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#acesso aos datasets
df_desmatamento = pd.read_csv('dataset/desmatamento_ucs.csv', encoding='latin1', sep=";")
df_especies = pd.read_csv('dataset/pda-especies_ameacadas.csv', encoding='latin1', sep=";")

#visao geral da base de dados
df_especies.describe()
df_desmatamento.describe()

'''
Tratamento e limpeza dos dados:
'''
#dataset de espécies
##limpeza
df_especies = df_especies[['Nome Científico', 'Nome Comum', 'Categoria', 'Grupo', 'Estado', 'Região', 'Bioma', 'Unidade de Conservação Federal']]
df_especies_sem_null = df_especies.dropna(subset='Nome Científico') #limpeza dos dados nulos

##tratamento das categorias repetidas / agrupamento
mascara_vulneravel = df_especies_sem_null['Categoria'] == 'Vulnerável (VU)'
df_especies_sem_null.loc[mascara_vulneravel, 'Categoria'] = 'Vulnerável'

mascara_em_perigo = df_especies_sem_null['Categoria'] == 'Em Perigo (EN)'
df_especies_sem_null.loc[mascara_em_perigo, 'Categoria'] = 'Em Perigo'

mascara_criticamente_em_perigo = df_especies_sem_null['Categoria'] == 'Criticamente em Perigo (CR)'
df_especies_sem_null.loc[mascara_criticamente_em_perigo, 'Categoria'] = 'Criticamente em Perigo'

mascara_extinta= (df_especies_sem_null['Categoria'] == 'Regionalmente Extinta') | (df_especies_sem_null['Categoria'] == 'Extinta na Natureza')
df_especies_sem_null.loc[mascara_extinta, 'Categoria'] = 'Extinta'

df_especies_tratado = df_especies_sem_null

df_especies_tratado

#tratamento base de desmantamento
##como a coluna "cnuc" não tem informação relevante, excluiremos da base.
df_desmatamento = df_desmatamento.drop("cnuc",axis=1)

##coletando os índices das linhas onde na coluna "prodes" existe valor "Não" e obedece demais comparações
index_names = df_desmatamento[ ((df_desmatamento['prodes'] == 'Não') & (df_desmatamento['2018'] == '12,5858')) | ((df_desmatamento['prodes'] == 'Não') & (df_desmatamento['2021'] == '33,4555')) | ((df_desmatamento['prodes'] == 'Não') & (df_desmatamento['2021'] == '3,2901'))].index
df_3linhas = df_desmatamento.iloc[index_names]
index_names = df_desmatamento[df_desmatamento['prodes'] == 'Não'].index
df_desmatamento.drop(index_names, inplace = True)
df_desmatamento = df_desmatamento.drop("prodes",axis=1)

##combinação dos dois dataframes
df_desmatamento = pd.concat([df_desmatamento, df_3linhas], ignore_index=True)

##renomeando a coluna 'uc' para 'Unidade de Conservação Federal'
df_desmatamento.rename(columns={'uc': 'Unidade de Conservação Federal'}, inplace=True)

df_desmatamento['2021'] = df_desmatamento['2021'].str.replace(',', '.')
df_desmatamento['2021'] = df_desmatamento['2021'].astype(np.float32)

'''
Análise exploratória dos dados por grupo de espécies:
'''
#distribuição das espécies por grupo
ax = sns.countplot(data=df_especies_tratado, y='Grupo', order=df_especies_tratado['Grupo'].value_counts().index.tolist())
ax.set_xlabel('Qtd de espécies')

'''
insights:
Com base no gráfico, é possível constatar que os grupos com maior incidência na lista de espécies ameaçadas 
são Peixes Continentais, Invertebrados Terrestres e Aves, respectivamente. Em contrapartida, os grupos menos preocupantes 
são os Mamíferos Aquáticos e Invertebrados Marinhos, respectivamente.
'''

#distribuição das espécies por grupo realicionadas ao risco de extinção
ax = sns.countplot(data=df_especies_tratado, y='Grupo', hue='Categoria', order=df_especies_tratado['Grupo'].value_counts().index.tolist())
ax.set_xlabel('Qtd de espécies')

'''
insights:
Com base no gráfico, é possível constatar que os únicos grupos que contém espécies extintas
 são: Aves, Anfíbios e Mamíferos, mas ainda assim com baixa incidência.'''

#distribuição de espécies vulneráveis por grupo.
df_vulneravel = df_especies_tratado[df_especies_tratado['Categoria'] == 'Vulnerável']
ax1 = sns.countplot(data=df_vulneravel, y='Grupo', order=df_vulneravel['Grupo'].value_counts().index.tolist())
ax1.set_xlabel('Qtd de espécies vulneráveis')

#distribuição de espécies em perigo por grupo
df_em_perigo = df_especies_tratado[df_especies_tratado['Categoria'] == 'Em Perigo']
ax2 = sns.countplot(data=df_em_perigo, y='Grupo', order=df_em_perigo['Grupo'].value_counts().index.tolist())
ax2.set_xlabel('Qtd de espécies em perigo')

#distribuição de espécies criticamente em perigo por grupo
df_criticamente_em_perigo = df_especies_tratado[df_especies_tratado['Categoria'] == 'Criticamente em Perigo']
ax3 = sns.countplot(data=df_criticamente_em_perigo, y='Grupo', order=df_criticamente_em_perigo['Grupo'].value_counts().index.tolist())
ax3.set_xlabel('Qtd de espécies criticamente em perigo')

#distribuição de espécies extintas por grupo
df_extinta = df_especies_tratado[df_especies_tratado['Categoria'] == 'Extinta']
ax4 = sns.countplot(data=df_extinta, y='Grupo', order=df_extinta['Grupo'].value_counts().index.tolist())
ax4.set_xlabel('Qtd de espécies extintas')

'''
Análise exploratória dos dados por categoria de risco de extinção:
'''
#distribuição de espécies por categoria de risco de extinção
sns.countplot(df_especies_tratado, y="Categoria", order = df_especies_tratado["Categoria"].value_counts().index.tolist())

'''
insights:
é possível observar que a população de espécies com um grau de risco de extinção mais elevado são menos numerosas
que as espécies com um grau de risco de extinção menos elevado.
'''

#distribuição de espécies por categoria de extinção em unidades de conservação federal
mascara_sem_uc = df_especies_tratado['Unidade de Conservação Federal'].isnull()
mascara_com_uc = (~mascara_sem_uc)

df_especies_com_coluna_ucf = df_especies_tratado
df_especies_com_coluna_ucf['Localizada em UCF'] = mascara_com_uc
df_especies_tratado.loc[mascara_com_uc, 'Localizada em UCF'] = 'Sim'
df_especies_tratado.loc[mascara_sem_uc, 'Localizada em UCF'] = 'Não'

df_especies_com_coluna_ucf['Localizada em UCF'].value_counts()

sns.countplot(df_especies_com_coluna_ucf, y="Categoria", hue='Localizada em UCF', order = df_especies_com_coluna_ucf["Categoria"].value_counts().index.tolist())

''' 
insights:

É possível observar não existe nenhuma espécie extinta que estava protegida em uma unidade de conservação federal
Além disso, é possivel notar que as espécies com um grau de risco de extinção mais elevado estão em menor quantidade nas unidades de conservação, se comparadas
as espécies que não estão protegidas por uma unidade de conservação
'''

'''
Análise exploratória dos dados por desmantamento em Unidades de Conservação Federal:
'''
#média de desmatamento por bioma
df_desmatamento.groupby('biocgfun')['2021'].mean().sort_values(ascending=False)
#calc a média agrupada por biocgfun
media_por_bioma = df_desmatamento.groupby('biocgfun')['2021'].mean().reset_index()


sns.barplot(x='2021', y='biocgfun', data=media_por_bioma, palette='viridis')

'''
Análise exploratória dos dados por espécies por bioma:
'''
#criação do dataframe que relaciona espécies por bioma
categoria_por_bioma = []

biomas = ['Amazônia', 'Caatinga', 'Cerrado', 'Marinho', 'Mata Atlântica', 'Pampa', 'Pantanal', 'Desconhecido']

for i in biomas:
    vulneravel_por_bioma = 0
    perigo_por_bioma = 0
    criticamente_em_perigo_por_bioma = 0
    extinta_por_bioma = 0

    index_linha = -1
    for j in df_especies_tratado['Bioma'].astype(str):
        index_linha = index_linha + 1
        if j.count(i) == 0:
            continue     
           
        if df_especies_tratado['Categoria'][index_linha] == 'Vulnerável':
            vulneravel_por_bioma = vulneravel_por_bioma + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Em Perigo':
            perigo_por_bioma = perigo_por_bioma + 1           
        elif df_especies_tratado['Categoria'][index_linha] == 'Criticamente em Perigo':
            criticamente_em_perigo_por_bioma = criticamente_em_perigo_por_bioma + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Extinta':
            extinta_por_bioma = extinta_por_bioma + 1
    quantidade_total = vulneravel_por_bioma + perigo_por_bioma + criticamente_em_perigo_por_bioma + extinta_por_bioma
    categoria_por_bioma.append([i, quantidade_total, vulneravel_por_bioma, perigo_por_bioma, criticamente_em_perigo_por_bioma, extinta_por_bioma])

df_biomas = pd.DataFrame(categoria_por_bioma, columns=['Bioma', 'Quantidade de Espécies', 'Vulnerável', 'Em Perigo', 'Criticamente em Perigo', 'Extinta'])

'''
insights:
Podemos observar que a quantidade de espécies em cada categoria é bem distribuida
descrescendo levemente ao aumentar o grau de perigo da categoria, exceto em
extinto que são poucos as espécies extintas. 
'''

#estatísticas do dataframe espécies por bioma
df_biomas.describe()

#boxplot da quantidade de espécies por bioma
sns.boxplot(df_biomas, x="Quantidade de Espécies", orient="h")

#gráfico da quantidade de espécies por bioma
df_biomas = df_biomas.sort_values(by='Quantidade de Espécies', ascending=False)
sns.barplot(df_biomas, x="Quantidade de Espécies", y="Bioma", estimator=np.median, errorbar=None)

#gráfico da quantidade de espécies na categoria vunerável por bioma
plt.figure()
df_biomas = df_biomas.sort_values(by='Vulnerável', ascending=False)
sns.barplot(df_biomas, x="Vulnerável", y="Bioma",errorbar=None)

#gráfico da quantidade de espécies na categoria em perigo por bioma
plt.figure()
df_biomas = df_biomas.sort_values(by='Em Perigo', ascending=False)
sns.barplot(df_biomas, x="Em Perigo", y="Bioma", errorbar=None)

#gráfico da quantidade de espécies na categoria criticamente em perigo por bioma
plt.figure()
df_biomas = df_biomas.sort_values(by='Criticamente em Perigo', ascending=False)
sns.barplot(df_biomas, x="Criticamente em Perigo", y="Bioma", errorbar=None)

#gráfico da quantidade de espécies na categoria extinta por bioma
plt.figure()
df_biomas = df_biomas.sort_values(by='Extinta', ascending=False)
sns.barplot(df_biomas, x="Extinta", y="Bioma",errorbar=None)

'''
Análise exploratória de dados por estado:
'''
#criação do dataframe que relaciona espécies por estado
categoria_por_estado = []

estados = ["Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará",
    "Distrito Federal", "Espírito Santo", "Goiás", "Maranhão",
    "Mato Grosso", "Mato Grosso do Sul", "Minas Gerais",
    "Pará", "Paraíba", "Paraná", "Pernambuco", "Piauí", "Rio de Janeiro",
    "Rio Grande do Norte", "Rio Grande do Sul", "Rondônia", "Roraima",
    "Santa Catarina", "São Paulo", "Sergipe", "Tocantins"]

for i in estados:
    vulneravel_por_estado = 0
    perigo_por_estado = 0
    criticamente_em_perigo_por_estado = 0
    extinta_por_estado = 0

    index_linha = -1
    for j in df_especies_tratado['Estado'].astype(str):
        index_linha = index_linha + 1
        if j.count(i) == 0:
            continue     
           
        if df_especies_tratado['Categoria'][index_linha] == 'Vulnerável':
            vulneravel_por_estado = vulneravel_por_estado + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Em Perigo':
            perigo_por_estado = perigo_por_estado + 1           
        elif df_especies_tratado['Categoria'][index_linha] == 'Criticamente em Perigo':
            criticamente_em_perigo_por_estado = criticamente_em_perigo_por_estado + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Extinta':
            extinta_por_estado = extinta_por_estado + 1
    quantidade_total = vulneravel_por_estado + perigo_por_estado + criticamente_em_perigo_por_estado + extinta_por_estado
    categoria_por_estado.append([i, quantidade_total, vulneravel_por_estado, perigo_por_estado, criticamente_em_perigo_por_estado, extinta_por_estado])

df_estados = pd.DataFrame(categoria_por_estado, columns=['Estado', 'Quantidade de Espécies', 'Vulnerável', 'Em Perigo', 'Criticamente em Perigo', 'Extinta'])
df_estados

#estatísticas do dataframe de espécies por estados
df_estados.describe()

#boxplot da quantidade de espécies por estado
sns.boxplot(df_estados, x="Quantidade de Espécies", orient="h")

#distribuição de quantidade de espécies por estado
df_estados = df_estados.sort_values(by='Quantidade de Espécies', ascending=False)
sns.barplot(df_estados, x="Quantidade de Espécies", y="Estado", errorbar=None)

'''
Insights:
é possível observar que Bahia, Minas Gerais e São Paulo são os estados com maior número de espécies em risco,
enquanto Acre, Roraima e Amapá são os estados com o menor número
'''

#distribuição de espécies consideradas vulneráveis por estado
df_estados = df_estados.sort_values(by='Vulnerável', ascending=False)
sns.barplot(df_estados, x="Vulnerável", y="Estado", errorbar=None)

'''
Insights:
é possível notar que Pará, Minas Gerais, Bahia e São Paulo são os estados com maior número de espécies consideradas vulneráveis.
Todos esses 4 estados estão no top5 dos que apresentam a maior quantidade de espécies.

enquanto Amapá, Sergipe e Distrito Federal são o que apresentam menor número
'''

#distribuição de espécies consideradas em perigo por estado
df_estados = df_estados.sort_values(by='Em Perigo', ascending=False)
sns.barplot(df_estados, x="Em Perigo", y="Estado", errorbar=None)

'''
insights:
é possível observar que Bahia, Minas Gerais e São Paulo são os estados com maior número de espécies em perigo,
esses três estados também pertence ao top3 geral

Acre, Amapá e Rondônia possuem os menores índices.

'''

#distribuição de espécies consideradas criticamente em perigo por estado
df_estados = df_estados.sort_values(by='Criticamente em Perigo', ascending=False)
sns.barplot(df_estados, x="Criticamente em Perigo", y="Estado", errorbar=None)

'''
insights:
é possível observar que Minas Gerais, São Paulo e Rio de Janeiro são os estados com maior número de espécies em risco,
eles também lideram o top5 dos espécies gerais

Acre, Rondônia e Roraima são os que apresentam menos
'''

#distribuição de espécies consideradas extintas por estado
df_estados = df_estados.sort_values(by='Extinta', ascending=False)
sns.barplot(df_estados, x="Extinta", y="Estado", errorbar=None)

'''
insights:
Só há espécies extintas em 8 estados Brasileiros, sendo eles: ...
São Paulo, mais uma vez, lidera o ranking.
'''

'''
Análise exploratória de dados por região:
'''
#criação novo dataframe por regioes
categoria_por_regiao = []

regioes = ["Sul", "Sudeste", "Centro-Oeste", "Nordeste", "Norte"]

for i in regioes:
    vulneravel_por_regiao = 0
    perigo_por_regiao = 0
    criticamente_em_perigo_por_regiao = 0
    extinta_por_regiao = 0

    index_linha = -1
    for j in df_especies_tratado['Região'].astype(str):
        index_linha = index_linha + 1
        if j.count(i) == 0:
            continue

        if df_especies_tratado['Categoria'][index_linha] == 'Vulnerável':
            vulneravel_por_regiao = vulneravel_por_regiao + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Em Perigo':
            perigo_por_regiao = perigo_por_regiao + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Criticamente em Perigo':
            criticamente_em_perigo_por_regiao = criticamente_em_perigo_por_regiao + 1
        elif df_especies_tratado['Categoria'][index_linha] == 'Extinta':
            extinta_por_regiao = extinta_por_regiao + 1
    quantidade_total = vulneravel_por_regiao + perigo_por_regiao + criticamente_em_perigo_por_regiao + extinta_por_regiao
    categoria_por_regiao.append([i, quantidade_total, vulneravel_por_regiao, perigo_por_regiao, criticamente_em_perigo_por_regiao, extinta_por_regiao])

df_regioes = pd.DataFrame(categoria_por_regiao, columns=['Região', 'Quantidade de Espécies', 'Vulnerável', 'Em Perigo', 'Criticamente em Perigo', 'Extinta'])
df_regioes

df_regioes = df_regioes.sort_values(by='Quantidade de Espécies', ascending=False)

df_regioes

#distribuição de espécies por região
sns.barplot(df_regioes, x="Quantidade de Espécies", y="Região", estimator=np.median, errorbar=None)

#distribuição de espécies vulneráveis por região
df_regioes = df_regioes.sort_values(by='Vulnerável', ascending=False)
sns.barplot(df_regioes, x="Vulnerável", y="Região", errorbar=None)

#distribuição de espécies em perigo por região
df_regioes = df_regioes.sort_values(by='Em Perigo', ascending=False)
sns.barplot(df_regioes, x="Em Perigo", y="Região", errorbar=None)

#distribuição de espécies criticamente em perigo por região
df_regioes = df_regioes.sort_values(by='Criticamente em Perigo', ascending=False)
sns.barplot(df_regioes, x="Criticamente em Perigo", y="Região",errorbar=None)

#distribuição de espécies consideradas extintas por região
df_regioes = df_regioes.sort_values(by='Extinta', ascending=False)
sns.barplot(df_regioes, x="Extinta", y="Região", errorbar=None)

'''Conclusões:
- O bioma mais desmatado do Brasil, a Mata Atlântica, não possui monitoramento em suas Unidades de Conservação. Em consonância, a Mata Atlântica é o bioma que apresenta mais espécies inlcuídas na lista de risco de extinção, assim como apresenta o maior número de espécies extintas;
- Não há espécies extintas em Unidades de Conservação Federais;
- A quantidade de espécies ameaçadas de extinção presentes em unidades de conservação ambiental é inferior, se compoaradas às não presentes em áreas protegidas;
- As espécies em risco mais expressivas são: peixes continentais, invertebrados terrestres e aves. Em contrapartida, as espécies menos presentes são: mamíferos aquáticos, invertebrados marinhos e anfíbios;
- O grupo com mais espécies extintas é o das aves;
- Os estados que mais apresentam espécies em risco são: Minas Gerais, Bahia e São Paulo, este último com o maior número de espécies extintas. Analogamente, sudeste e nordeste são as regiões mais significativas em termos de espécies ameaçadas.'''