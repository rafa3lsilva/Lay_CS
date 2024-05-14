import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit as st
import datetime


def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, num_simulations=10000, random_seed=42):
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

def top_results_df(simulated_results, top_n):
    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']
    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts
    return result_counts

st.set_page_config(
    page_title="Apostas Lay para os Resultados Selecionados",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

dia = st.sidebar.date_input("Selecione a data:", value=datetime.date.today(), key='date_input')
st.markdown("""<style>
    div[data-testid="stDateInput"] > div:first-child {
        width: 40px;
    }
</style>""", unsafe_allow_html=True)

leagues = ['ARGENTINA - LIGA PROFESIONAL', 'ARGENTINA - COPA DE LA LIGA PROFESIONAL', 'ARMENIA - PREMIER LEAGUE', 'AUSTRALIA - A-LEAGUE', 'AUSTRIA - 2. LIGA', 'AUSTRIA - BUNDESLIGA', 'BELGIUM - CHALLENGER PRO LEAGUE', 'BELGIUM - JUPILER PRO LEAGUE', 'BOSNIA AND HERZEGOVINA - PREMIJER LIGA BIH', 'BRAZIL - COPA DO BRASIL', 'BRAZIL - SERIE A', 'BRAZIL - SERIE B', 'BULGARIA - PARVA LIGA', 'CHINA - SUPER LEAGUE', 'CROATIA - HNL', 'CROATIA - PRVA NL', 'CZECH REPUBLIC - FORTUNA:LIGA', 'DENMARK - 1ST DIVISION', 'DENMARK - SUPERLIGA', 'EGYPT - PREMIER LEAGUE', 'ENGLAND - CHAMPIONSHIP', 'ENGLAND - LEAGUE ONE', 'ENGLAND - LEAGUE TWO', 'ENGLAND - NATIONAL LEAGUE', 'ENGLAND - PREMIER LEAGUE', 'ESTONIA - ESILIIGA', 'ESTONIA - MEISTRILIIGA', 'EUROPE - CHAMPIONS LEAGUE', 'EUROPE - EUROPA CONFERENCE LEAGUE', 'EUROPE - EUROPA LEAGUE', 'FINLAND - VEIKKAUSLIIGA', 'FINLAND - YKKONEN', 'FRANCE - LIGUE 1', 'FRANCE - LIGUE 2', 'FRANCE - NATIONAL', 'GERMANY - 2. BUNDESLIGA', 'GERMANY - 3. LIGA', 'GERMANY - BUNDESLIGA', 'HUNGARY - OTP BANK LIGA', 'ICELAND - BESTA DEILD KARLA', 'IRELAND - PREMIER DIVISION', 'ITALY - SERIE A', 'ITALY - SERIE B', 'JAPAN - J1 LEAGUE', 'JAPAN - J2 LEAGUE', 'MEXICO - LIGA DE EXPANSION MX', 'MEXICO - LIGA MX', 'NETHERLANDS - EERSTE DIVISIE', 'NETHERLANDS - EREDIVISIE', 'NORWAY - ELITESERIEN', 'NORWAY - OBOS-LIGAEN', 'POLAND - EKSTRAKLASA', 'PORTUGAL - LIGA PORTUGAL', 'PORTUGAL - LIGA PORTUGAL 2', 'ROMANIA - LIGA 1', 'SAUDI ARABIA - SAUDI PROFESSIONAL LEAGUE', 'SCOTLAND - CHAMPIONSHIP', 'SCOTLAND - LEAGUE ONE', 'SCOTLAND - LEAGUE TWO', 'SCOTLAND - PREMIERSHIP', 'SERBIA - SUPER LIGA', 'SLOVAKIA - NIKE LIGA', 'SLOVENIA - PRVA LIGA', 'SOUTH AMERICA - COPA LIBERTADORES', 'SOUTH AMERICA - COPA SUDAMERICANA', 'SOUTH KOREA - K LEAGUE 1', 'SOUTH KOREA - K LEAGUE 2', 'SPAIN - LALIGA', 'SPAIN - LALIGA2', 'SWEDEN - ALLSVENSKAN', 'SWEDEN - SUPERETTAN', 'SWITZERLAND - CHALLENGE LEAGUE', 'SWITZERLAND - SUPER LEAGUE', 'TURKEY - 1. LIG', 'TURKEY - SUPER LIG', 'UKRAINE - PREMIER LEAGUE', 'USA - MLS', 'WALES - CYMRU PREMIER']

jogos_do_dia = pd.read_csv(f'https://github.com/futpythontrader/YouTube/blob/main/Jogos_do_Dia/FlashScore/Jogos_do_Dia_FlashScore_{dia}.csv?raw=true')
jogos_do_dia = jogos_do_dia[['League','Date','Time','Home','Away','Odd_H','Odd_D','Odd_A']]
jogos_do_dia.columns = ['League','Date','Time','Home','Away','Odd_H','Odd_D','Odd_A']
Jogos_do_Dia = drop_reset_index(jogos_do_dia)

Jogos = Jogos_do_Dia.sort_values(by='League')
ligas = Jogos['League'].unique()

base = pd.read_csv("https://github.com/futpythontrader/YouTube/blob/main/Bases_de_Dados/FlashScore/Base_de_Dados_FlashScore_v2.csv?raw=true")
base = base[['League','Date','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A']]
base.columns = ['League','Date','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A']
base = base[base['League'].isin(ligas) == True]
base = drop_reset_index(base)

n = 5
base['Media_GM_H'] = base.groupby('Home')['Goals_H'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GM_A'] = base.groupby('Away')['Goals_A'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GM_H'] = base.groupby('Home')['Media_GM_H'].shift(1)
base['Media_GM_A'] = base.groupby('Away')['Media_GM_A'].shift(1)
base['Media_GS_H'] = base.groupby('Home')['Goals_A'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GS_A'] = base.groupby('Away')['Goals_H'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GS_H'] = base.groupby('Home')['Media_GS_H'].shift(1)
base['Media_GS_A'] = base.groupby('Away')['Media_GS_A'].shift(1)
base = drop_reset_index(base)

base_H = base[['Home','Media_GM_H','Media_GS_H']]
base_A = base[['Away','Media_GM_A','Media_GS_A']]

Jogos_do_Dia['Jogo'] = Jogos_do_Dia['Home'] + ' x ' + Jogos_do_Dia['Away']
lista_confrontos = Jogos_do_Dia['Jogo'].tolist()

st.header('Método de Lay CS')
st.text('Simulação de Poisson')
st.sidebar.title('Selecione a Partida')
wid_filtro = st.sidebar.selectbox('Escolha o Jogo:', Jogos_do_Dia['Jogo'].unique(), index=0)
df_filtrado = Jogos_do_Dia[Jogos_do_Dia['Jogo'] == wid_filtro]
df_filtrado = df_filtrado[['League','Date','Time','Home','Away','Odd_H','Odd_D','Odd_A']]
df_filtrado = drop_reset_index(df_filtrado)
st.write('**Jogo Selecionado:**')
st.write(df_filtrado)

# Variáveis globais para armazenar as informações do jogo selecionado

Team_01 = df_filtrado['Home'].iloc[0]
Team_02 = df_filtrado['Away'].iloc[0]
Time = df_filtrado['Time'].iloc[0]
Date = df_filtrado['Date'].iloc[0]
Media_GM_H = base_H.loc[base_H['Home'] == Team_01, 'Media_GM_H'].iloc[0]
Media_GM_A = base_A.loc[base_A['Away'] == Team_02, 'Media_GM_A'].iloc[0]
Media_GS_H = base_H.loc[base_H['Home'] == Team_01, 'Media_GS_H'].iloc[0]
Media_GS_A = base_A.loc[base_A['Away'] == Team_02, 'Media_GS_A'].iloc[0]

simulated_results = simulate_match(Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
results = top_results_df(simulated_results, 10000)

# Defina a probabilidade máxima desejada
probabilidade_maxima = 0.08 

# Filtrar os resultados com probabilidade igual ou inferior a 8%
results_filtrado = results[results['Probability'] < probabilidade_maxima]

def format_score(row):
    home_goals = int(row['Home_Goals'])
    away_goals = int(row['Away_Goals'])

    # Verificar se é uma goleada para o time da casa
    if home_goals > 3 and home_goals > away_goals:
        return 'Goleada_H'
    
    # Verificar se é uma goleada para o time visitante
    elif away_goals > 3 and away_goals > home_goals:
        return 'Goleada_A'
    
    # Verificar se é um empate com muitos gols
    elif home_goals >= 4 and away_goals >= 4 and home_goals == away_goals:
        return 'Outro_D'

    # Se não for nenhuma das condições acima, retornar o placar normal
    else:
        return f"{home_goals}x{away_goals}"

results_filtrado['Placar'] = results_filtrado.apply(format_score, axis=1)

# Calcular e adicionar a coluna 'Odd_Justa' com a odd justa para cada placar
results_filtrado['Odd_Justa'] = 1 / results_filtrado['Probability']

# Remover as colunas 'Home_Goals' e 'Away_Goals'
results_filtrado = results_filtrado[['Placar', 'Count', 'Probability', 'Odd_Justa']]

# Arredondar 'Probability' e 'Odd_Justa' para duas casas decimais
results_filtrado['Probability'] = results_filtrado['Probability'].round(2)
results_filtrado['Odd_Justa'] = results_filtrado['Odd_Justa'].round(2)

results_filtrado = results_filtrado.head(5)
results_filtrado = results_filtrado.reset_index(drop=True)
results_filtrado.index += 1

# Exibir os resultados filtrados
last_home_games = base[base['Home'] == df_filtrado['Home'].iloc[0]].tail(5).reset_index(drop=True)
last_home_games.index += 1
last_away_games = base[base['Away'] == df_filtrado['Away'].iloc[0]].tail(5).reset_index(drop=True)
last_away_games.index += 1

st.write('**Últimos 5 jogos do time da casa:**')
st.write(last_home_games)

st.write('**Últimos 5 jogos do time visitante:**')
st.write(last_away_games)

# Cabeçalho da seção

# Criação do DataFrame novo_df com base no results_filtrado
novo_df = results_filtrado.copy()

# Adição do input para responsabilidade desejada
responsabilidade_desejada = st.sidebar.number_input("Responsabilidade Desejada:", min_value=0.0, step=1.0)

# Adição das odds de mercado ao novo_df
odds_mercado = []
for index, row in novo_df.iterrows():
    odd_mercado = st.sidebar.number_input(f"Odd de mercado para '{row['Placar']}':", min_value=1.01, step=0.01)
    odds_mercado.append(odd_mercado)

# Adição das odds de mercado ao novo_df
novo_df['Odd_Mercado'] = odds_mercado

# Função para calcular o tamanho da stake com base na responsabilidade desejada
def calcular_tamanho_stake(responsabilidade, odd_justa, odd_mercado):
    return responsabilidade / (odd_mercado - 1)

# Aplicação do cálculo do tamanho da stake
novo_df['Tamanho_Stake'] = calcular_tamanho_stake(responsabilidade_desejada, novo_df['Odd_Justa'], novo_df['Odd_Mercado'])

# Cálculo do lucro potencial
comissao = 5.6 / 100  # Comissão de 5.6%
novo_df['Lucro_Potencial'] = novo_df['Tamanho_Stake'] * (1 - comissao)

# Função para calcular o tamanho da stake com base no critério de Kelly
def calcular_tamanho_stake_kelly(probabilidade, odd_justa, odd_mercado):
    if odd_mercado < odd_justa:  # Aposta de valor apenas se a odd de mercado for menor que a odd justa
        kelly_fraction = (probabilidade * odd_justa - 1) / (odd_mercado - 1)
        return kelly_fraction
    else:
        return 0  # Se a odd de mercado for maior ou igual à odd justa, não há aposta de valor

# Aplicação do cálculo do tamanho da stake de acordo com o critério de Kelly
novo_df['Tamanho_Stake_Kelly'] = novo_df.apply(lambda row: calcular_tamanho_stake_kelly(row['Probability'], row['Odd_Justa'], row['Odd_Mercado']), axis=1)

# Arredondar Tamanho_Stake, Lucro_Potencial e Tamanho_Stake_Kelly para duas casas decimais
novo_df['Tamanho_Stake'] = novo_df['Tamanho_Stake'].round(2)
novo_df['Lucro_Potencial'] = novo_df['Lucro_Potencial'].round(2)
novo_df['Tamanho_Stake_Kelly'] = novo_df['Tamanho_Stake_Kelly'].round(2)

# Verificar se a aposta tem EV+ usando o critério de Kelly
novo_df['EV_Kelly'] = novo_df['Tamanho_Stake_Kelly'] > 0

# Exibição do resultado final
st.write('Resultados com tamanhos de stake para alcançar a responsabilidade desejada e critério de Kelly:')
st.write(novo_df[['Placar','Count', 'Probability', 'Odd_Justa', 'Odd_Mercado', 'Tamanho_Stake',
                  'Lucro_Potencial', 'Tamanho_Stake_Kelly', 'EV_Kelly']])
