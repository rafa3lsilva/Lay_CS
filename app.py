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

st.write('Resultados da Analise')

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

# Adicionar a coluna 'Placar' com o formato 'Home_Goals x Away_Goals'
results_filtrado['Placar'] = results_filtrado.apply(lambda row: f"{int(row['Home_Goals'])}x{int(row['Away_Goals'])}", axis=1)

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
st.write(results_filtrado)

# Cabeçalho da seção
# st.subheader('Apostas Lay para os Resultados Selecionados')

# Defina a responsabilidade total
responsabilidade_total = st.sidebar.number_input("Responsabilidade:", min_value=1.01, step=0.01)

# Defina a comissão
comissao = 5.6 / 100  # 5.6% de comissão

# Lista para armazenar as odds de mercado
odds_mercado = []

# Loop sobre cada placar para coletar as odds de mercado
for placar in results_filtrado['Placar']:
    odd_mercado = st.sidebar.number_input(f"Odd de mercado para '{placar}':", min_value=1.01, step=0.01)
    odds_mercado.append(odd_mercado)

# Adicione as odds de mercado ao DataFrame
results_filtrado['Odd_Mercado'] = odds_mercado

# Calcular a stake para cada placar com base na responsabilidade total
stakes = [responsabilidade_total / (odd - 1) for odd in odds_mercado]
results_filtrado['Stake'] = stakes

# Arredondar a stake para duas casas decimais
results_filtrado['Stake'] = results_filtrado['Stake'].round(2)

# Calcular o lucro potencial para cada placar
lucros_potenciais = [stake * (1 - comissao) for stake in stakes]
results_filtrado['Lucro_Potencial'] = lucros_potenciais

# Arredondar o lucro potencial para duas casas decimais
results_filtrado['Lucro_Potencial'] = results_filtrado['Lucro_Potencial'].round(2)

# % de lucro sobre a responsabilidade
porcentagem_lucro = [lucros_potenciais / responsabilidade_total * 100 for lucros_potenciais in lucros_potenciais]
results_filtrado['%Lucro'] = porcentagem_lucro

# Arredondar o lucro potencial para duas casas decimais
results_filtrado['%Lucro'] = results_filtrado['%Lucro'].round(2)

# Calcular o EV para cada placar comparando a odd de mercado com a odd justa
ev = []
for index, row in results_filtrado.iterrows():
    ev_placar = (1 - comissao) * (row['Odd_Justa'] - 1) - (1 - comissao) * (row['Odd_Mercado'] - 1)
    ev.append(ev_placar)

# Arredondar a odd justa para duas casas decimais
results_filtrado['Odd_Justa'] = results_filtrado['Odd_Justa'].round(2)

results_filtrado['EV'] = ev

# Verificar se o EV é positivo
results_filtrado['EV+'] = results_filtrado['EV'] > 0

# Arredondar o EV para duas casas decimais
results_filtrado['EV'] = results_filtrado['EV'].round(2)


# Filtrar os resultados onde o EV é positivo
results_filtrado_ev_positivo = results_filtrado[results_filtrado['EV+']]

results_filtrado_ev_positivo = results_filtrado_ev_positivo.reset_index(drop=True)
results_filtrado_ev_positivo.index += 1

# Exibir os resultados
st.write('Resultados com EV positivo:')
st.write(results_filtrado_ev_positivo)

# Calcular o EV médio
EV_medio = results_filtrado_ev_positivo['EV'].mean()
st.write(f'EV médio: {round(EV_medio, 2)}')

