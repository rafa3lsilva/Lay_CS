import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit as st

# Função para simular os jogos
def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against,
                   num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)
    estimated_home_goals = (home_goals_for + away_goals_against) / 2
    estimated_away_goals = (away_goals_for + home_goals_against) / 2

    home_goals = poisson(estimated_home_goals).rvs(num_simulations)
    away_goals = poisson(estimated_away_goals).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results

# Função para verificar os resultados mais comuns das simulações de partida.
def top_results_df(simulated_results, top_n):
    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

# Função para calcular as médias de gols das equipes
def calcular_medias_gols(df, equipe_coluna, gols_coluna, janela=5):
    df['Media_Gols'] = df.groupby(equipe_coluna)[gols_coluna].rolling(window=janela, min_periods=1).mean().reset_index(0, drop=True)
    df['Media_Gols'] = df.groupby(equipe_coluna)['Media_Gols'].shift(1)
    return df

# Função para verificar valores ausentes nos dados
def verificar_valores_ausentes(df):
    return df.isnull().sum().sum() == 0

# Função para simular uma partida de futebol
def simular_partida(media_gols_casa, media_gols_fora, num_simulacoes=10000, seed=42):
    np.random.seed(seed)
    gols_casa = poisson(media_gols_casa).rvs(num_simulacoes)
    gols_fora = poisson(media_gols_fora).rvs(num_simulacoes)
    return gols_casa, gols_fora

# Função para calcular o lucro potencial com base na stake, odd e comissão da casa de apostas
def calcular_lucro_potencial(stake, odd, comissao):
    return (stake * (odd - 1)) * (1 - comissao)

# Função para calcular o tamanho da stake com base na responsabilidade desejada
def calcular_stake(responsabilidade, odd):
    return responsabilidade / (odd - 1)

# Função para verificar se uma aposta tem valor com base no critério de Kelly
def verificar_valor_kelly(probabilidade, odd_justa, odd_mercado):
    return (probabilidade * odd_justa - 1) / (odd_mercado - 1)

# Função para formatar as probabilidades de Kelly como porcentagem
def formatar_kelly_porcentagem(kelly):
    return '{:.2f}%'.format(kelly * 100)

# Função para verificar se uma aposta tem valor com base no critério de Kelly
def verificar_valor_kelly_evp(kelly):
    return kelly > 0

# Função para calcular a média móvel das probabilidades de um DataFrame
def calcular_media_movel_probabilidades(df, janela=5):
    df['Probabilidade_Media_Movel'] = df['Probability'].rolling(window=janela, min_periods=1).mean()
    return df

def drop_reset_index(df):
    df = df.dropna()  # Remove linhas com valores ausentes
    # Redefine os índices, descartando os antigos
    df = df.reset_index(drop=True)
    df.index += 1  # Adiciona 1 aos índices
    return df

# Carregar os dados
@st.cache_data
def carregar_dados(file_path):
    return pd.read_csv(file_path)

st.set_page_config(
    page_title="CS Statistic",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title('CS Statistic')

dia = st.sidebar.date_input(
    "Selecione a data:", value=datetime.date.today(), key='date_input')
st.markdown("""<style>
    div[data-testid="stDateInput"] > div:first-child {
        width: 50px;
    }
</style>""", unsafe_allow_html=True)

st.title('CS Statistic')

leagues = ['ARGENTINA - LIGA PROFESIONAL', 'ARGENTINA - COPA DE LA LIGA PROFESIONAL',
           'ARMENIA - PREMIER LEAGUE', 'AUSTRALIA - A-LEAGUE', 'AUSTRIA - 2. LIGA', 'AUSTRIA - BUNDESLIGA',
           'BELGIUM - CHALLENGER PRO LEAGUE', 'BELGIUM - JUPILER PRO LEAGUE', 'BOSNIA AND HERZEGOVINA - PREMIJER LIGA BIH',
           'BRAZIL - COPA DO BRASIL', 'BRAZIL - SERIE A', 'BRAZIL - SERIE B', 'BULGARIA - PARVA LIGA',
           'CHINA - SUPER LEAGUE', 'CROATIA - HNL', 'CROATIA - PRVA NL', 'CZECH REPUBLIC - FORTUNA:LIGA',
           'DENMARK - 1ST DIVISION', 'DENMARK - SUPERLIGA', 'EGYPT - PREMIER LEAGUE', 'ENGLAND - CHAMPIONSHIP',
           'ENGLAND - LEAGUE ONE', 'ENGLAND - LEAGUE TWO', 'ENGLAND - NATIONAL LEAGUE', 'ENGLAND - PREMIER LEAGUE',
           'ESTONIA - ESILIIGA', 'ESTONIA - MEISTRILIIGA', 'EUROPE - CHAMPIONS LEAGUE', 'EUROPE - EUROPA CONFERENCE LEAGUE',
           'EUROPE - EUROPA LEAGUE', 'FINLAND - VEIKKAUSLIIGA', 'FINLAND - YKKONEN', 'FRANCE - LIGUE 1', 'FRANCE - LIGUE 2',
           'FRANCE - NATIONAL', 'GERMANY - 2. BUNDESLIGA', 'GERMANY - 3. LIGA', 'GERMANY - BUNDESLIGA',
           'HUNGARY - OTP BANK LIGA', 'ICELAND - BESTA DEILD KARLA', 'IRELAND - PREMIER DIVISION', 'ITALY - SERIE A',
           'ITALY - SERIE B', 'JAPAN - J1 LEAGUE', 'JAPAN - J2 LEAGUE', 'MEXICO - LIGA DE EXPANSION MX', 'MEXICO - LIGA MX',
           'NETHERLANDS - EERSTE DIVISIE', 'NETHERLANDS - EREDIVISIE', 'NORWAY - ELITESERIEN', 'NORWAY - OBOS-LIGAEN',
           'POLAND - EKSTRAKLASA', 'PORTUGAL - LIGA PORTUGAL', 'PORTUGAL - LIGA PORTUGAL 2', 'ROMANIA - LIGA 1',
           'SAUDI ARABIA - SAUDI PROFESSIONAL LEAGUE', 'SCOTLAND - CHAMPIONSHIP', 'SCOTLAND - LEAGUE ONE', 'SCOTLAND - LEAGUE TWO',
           'SCOTLAND - PREMIERSHIP', 'SERBIA - SUPER LIGA', 'SLOVAKIA - NIKE LIGA', 'SLOVENIA - PRVA LIGA',
           'SOUTH AMERICA - COPA LIBERTADORES', 'SOUTH AMERICA - COPA SUDAMERICANA', 'SOUTH KOREA - K LEAGUE 1',
           'SOUTH KOREA - K LEAGUE 2', 'SPAIN - LALIGA', 'SPAIN - LALIGA2', 'SWEDEN - ALLSVENSKAN', 'SWEDEN - SUPERETTAN',
           'SWITZERLAND - CHALLENGE LEAGUE', 'SWITZERLAND - SUPER LEAGUE', 'TURKEY - 1. LIG', 'TURKEY - SUPER LIG',
           'UKRAINE - PREMIER LEAGUE', 'USA - MLS', 'WALES - CYMRU PREMIER']

# Importando os Jogos do Dia e a Base de Dados
jogos_do_dia = carregar_dados(
        f'https://github.com/futpythontrader/YouTube/blob/main/Jogos_do_Dia/FlashScore/Jogos_do_Dia_FlashScore_{dia}.csv?raw=true')
jogos_do_dia = jogos_do_dia[['League', 'Date',
                                'Time', 'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']]
jogos_do_dia.columns = ['League', 'Date', 'Time',
                            'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']
Jogos_do_Dia = jogos_do_dia[jogos_do_dia['League'].isin(leagues) == True]
Jogos_do_Dia = drop_reset_index(Jogos_do_Dia)

Jogos = Jogos_do_Dia.sort_values(by='League')
ligas = Jogos['League'].unique()

base = carregar_dados(
    "https://github.com/futpythontrader/YouTube/blob/main/Bases_de_Dados/FlashScore/Base_de_Dados_FlashScore_v2.csv?raw=true")
base = base[['League', 'Date', 'Home', 'Away',
             'Goals_H', 'Goals_A', 'Odd_H', 'Odd_D', 'Odd_A']]
base.columns = ['League', 'Date', 'Home', 'Away',
                'Goals_H', 'Goals_A', 'Odd_H', 'Odd_D', 'Odd_A']
base = base[base['League'].isin(ligas) == True]
base = drop_reset_index(base)

n = 5

# Médias Home
base['Media_GM_H'] = base.groupby('Home')['Goals_H'].rolling(
    window=n, min_periods=n).mean().reset_index(0, drop=True)
base['Media_GM_A'] = base.groupby('Away')['Goals_A'].rolling(
    window=n, min_periods=n).mean().reset_index(0, drop=True)
base['Media_GM_H'] = base.groupby('Home')['Media_GM_H'].shift(1)
base['Media_GM_A'] = base.groupby('Away')['Media_GM_A'].shift(1)
base = drop_reset_index(base)

# Médias Away
base['Media_GS_H'] = base.groupby('Home')['Goals_A'].rolling(
    window=n, min_periods=n).mean().reset_index(0, drop=True)
base['Media_GS_A'] = base.groupby('Away')['Goals_H'].rolling(
    window=n, min_periods=n).mean().reset_index(0, drop=True)
base['Media_GS_H'] = base.groupby('Home')['Media_GS_H'].shift(1)
base['Media_GS_A'] = base.groupby('Away')['Media_GS_A'].shift(1)
base = drop_reset_index(base)

base_H = base[['Home', 'Media_GM_H', 'Media_GS_H']]
base_A = base[['Away', 'Media_GM_A', 'Media_GS_A']]

# Criando a Lista com os Confrontos
Jogos_do_Dia['Jogo'] = Jogos_do_Dia['Home'] + ' x ' + Jogos_do_Dia['Away']
lista_confrontos = Jogos_do_Dia['Jogo'].tolist()

last_base_H = base_H.groupby('Home').last().reset_index()
last_base_A = base_A.groupby('Away').last().reset_index()

df = pd.merge(Jogos_do_Dia, last_base_H, how='left',
              left_on='Home', right_on='Home')
df = pd.merge(df, last_base_A, how='left', left_on='Away', right_on='Away')

# Seleção dos Jogos
wid_filtro = st.sidebar.selectbox(
    'Selecione o Horario:', Jogos_do_Dia['Time'].unique(), index=0)
df_Hora = Jogos_do_Dia[Jogos_do_Dia['Time'] == wid_filtro]
df_Hora = df_Hora[['League', 'Date', 'Time', 'Home',
                   'Away', 'Odd_H', 'Odd_D', 'Odd_A', 'Jogo']]

# Filtrar o DataFrame com base na liga selecionada
wid_filtro = st.sidebar.selectbox(
    'Selecione a Liga:', df_Hora['League'].unique(), index=0)
df_Liga = df_Hora[df_Hora['League'] == wid_filtro]
df_Liga = df_Liga[['League', 'Date', 'Time', 'Home',
                   'Away', 'Odd_H', 'Odd_D', 'Odd_A', 'Jogo']]

wid_filtro = st.sidebar.selectbox(
    'Escolha o Jogo:', df_Liga['Jogo'].unique(), index = 0)
df_filtrado = df_Liga[df_Liga['Jogo'] == wid_filtro]
df_filtrado = df_filtrado[['League', 'Date', 'Time',
                           'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']]

df_filtrado = drop_reset_index(df_filtrado)

st.write('**Jogo Selecionado:**')
st.write(df_filtrado)

# usando a simulação de poisson para definir os placares!!

i = 1

try:
    Liga = df.loc[i]['League']
    Team_01 = df.loc[i]['Home']
    Team_02 = df.loc[i]['Away']
    Time = df.loc[i]['Time']
    Date = df.loc[i]['Date']

except KeyError as e:
    # Caso ocorra o erro KeyError: 1, exibir uma mensagem de erro ou tomar outra ação apropriada
    st.error("Home e/ou Away com jogos inferior a 5.")
    # Você também pode adicionar mais detalhes sobre o erro se necessário
    st.error(f"Detalhes do erro: {e}")

Media_GM_H = df.loc[i]['Media_GM_H']
Media_GM_A = df.loc[i]['Media_GM_A']

Media_GS_H = df.loc[i]['Media_GS_H']
Media_GS_A = df.loc[i]['Media_GS_A']

simulated_results = simulate_match(
    Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
simulated_results = drop_reset_index(simulated_results)

results = top_results_df(simulated_results, 10000)
results = drop_reset_index(results)

results['Placar'] = results.apply(
    lambda row: 'Goleada_H' if (row['Home_Goals'] >= 4 and row['Home_Goals'] > row['Away_Goals'])
    else 'Goleada_A' if (row['Away_Goals'] >= 4 and row['Away_Goals'] > row['Home_Goals'])
    else 'Goleada_D' if (row['Home_Goals'] >= 4 and row['Away_Goals'] >= 4 and row['Home_Goals'] == row['Away_Goals'])
    else f"{int(row['Home_Goals'])}x{int(row['Away_Goals'])}", axis = 1
)

probabilidade_maxima = 0.08  # Defina a probabilidade mínima desejada

results_filtrado = results[results['Probability'] < probabilidade_maxima]

# Arredondar a odd justa para duas casas decimais
results_filtrado['Probability'] = results_filtrado['Probability'].round(2)

# Calcular as odds justas para os placares selecionados
results_filtrado['Odd_Justa'] = 1 / results_filtrado['Probability']

# Arredondar a odd justa para duas casas decimais
results_filtrado['Odd_Justa'] = results_filtrado['Odd_Justa'].round(2)

results_filtrado = results_filtrado.head(5)

results_filtrado = results_filtrado.reset_index(drop=True)
results_filtrado.index += 1

# Exibir os resultados filtrados
# Verifica se base e df_filtrado não estão vazios
if not base.empty and not df_filtrado.empty:
    # Verifica se há pelo menos um valor em 'Home' e 'Away' em df_filtrado
    if len(df_filtrado) > 0 and 'Home' in df_filtrado.columns and 'Away' in df_filtrado.columns:
        # Seleciona os últimos 5 jogos em casa para o time específico
        last_home_games = base[base['Home'] ==
                               df_filtrado['Home'].iloc[0]].tail(5)

        # Seleciona os últimos 5 jogos fora para o time específico
        last_away_games = base[base['Away'] ==
                               df_filtrado['Away'].iloc[0]].tail(5)
    else:
        print("Home e/ou Away com jogos inferior a 5.")
else:
    print("Home e/ou Away com jogos inferior a 5.")

# Classificar em ordem decrescente:
last_home_games = last_home_games.sort_values(
    by='Date', ascending=False).reset_index(drop=True)
last_home_games.index += 1

last_away_games = last_away_games.sort_values(
    by='Date', ascending=False).reset_index(drop=True)
last_away_games.index += 1

st.write('**Últimos 5 jogos do time da casa:**')
st.write(last_home_games)

st.write('**Últimos 5 jogos do time visitante:**')
st.write(last_away_games)

# Cabeçalho da seção
# Criação do DataFrame novo_df com base no results_filtrado
novo_df = results_filtrado.copy()

# Adição do input para responsabilidade desejada
responsabilidade_desejada = st.sidebar.number_input(
    "Responsabilidade:", min_value=0.0, step=1.0)

# Adicionar controle deslizante para escolher a fração de Kelly
kelly_fracao = st.sidebar.selectbox(
    "Selecione a fração de Kelly:",
    ["Kelly Completo", "Meio Kelly", "Um Quarto de Kelly"]
)

# Adição das odds de mercado ao novo_df
odds_mercado = []
for index, row in novo_df.iterrows():
    odd_mercado = st.sidebar.number_input(
        f"Odd Lay '{row['Placar']}':", min_value=1.01, step=0.01)
    odds_mercado.append(odd_mercado)

# Adição das odds de mercado ao novo_df
novo_df['Odd_Lay'] = odds_mercado

# Função para calcular o tamanho da stake com base na responsabilidade desejada
@st.cache_data
def calcular_tamanho_stake(responsabilidade, odd_justa, odd_mercado):
    return responsabilidade / (odd_mercado - 1)

# Aplicação do cálculo do tamanho da stake
novo_df['Stake'] = calcular_tamanho_stake(
    responsabilidade_desejada, novo_df['Odd_Justa'], novo_df['Odd_Lay'])

# Cálculo do lucro potencial
comissao = 5.6 / 100  # Comissão de 5.6%
novo_df['Lucro_Potencial'] = novo_df['Stake'] * (1 - comissao)

# Função para calcular o tamanho da stake com base na fração de Kelly selecionada
@st.cache_data
def calcular_tamanho_stake_kelly(probabilidade, odd_justa, odd_mercado, kelly_fracao):
    if kelly_fracao == "Kelly Completo":
        return (probabilidade * odd_justa - 1) / (odd_mercado - 1)
    elif kelly_fracao == "Meio Kelly":
        return 0.5 * (probabilidade * odd_justa - 1) / (odd_mercado - 1)
    elif kelly_fracao == "Um Quarto de Kelly":
        return 0.25 * (probabilidade * odd_justa - 1) / (odd_mercado - 1)

# Aplicar o cálculo do tamanho da stake com base na fração de Kelly selecionada
novo_df['Porc_Kelly'] = novo_df.apply(lambda row: calcular_tamanho_stake_kelly(
    row['Probability'], row['Odd_Justa'], row['Odd_Lay'], kelly_fracao), axis=1)

# Arredondar valores
novo_df['Stake'] = novo_df['Stake'].round(2)
novo_df['Lucro_Potencial'] = novo_df['Lucro_Potencial'].round(2)
novo_df['Porc_Kelly'] = (novo_df['Porc_Kelly'] * 100).round(4)

# Convertendo para porcentagem
novo_df['Porc_Kelly'] = novo_df['Porc_Kelly'] * 100

# Formatar os valores como porcentagem com duas casas decimais
novo_df['Porc_Kelly'] = novo_df['Porc_Kelly'].apply(
    lambda x: '{:.2f}%'.format(x))

# Verificar se a aposta tem EV+ usando o critério de Kelly
novo_df['EV_Kelly'] = novo_df['Porc_Kelly'].str.rstrip('%').astype(float) > 0

# Exibição do resultado final
st.write('Resultados com critério de Kelly:')
st.write(novo_df[['Placar', 'Count', 'Probability', 'Odd_Justa',
         'Odd_Lay', 'Stake', 'Lucro_Potencial', 'Porc_Kelly', 'EV_Kelly']])
