import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit as st
from urllib.error import HTTPError

# Função para resetar o index e iniciar em 1!!


@st.cache_data
def drop_reset_index(df):
    df = df.dropna()  # Remove linhas com valores ausentes
    # Redefine os índices, descartando os antigos
    df = df.reset_index(drop=True)
    df.index += 1  # Adiciona 1 aos índices
    return df

# Função para simular os jogos!!


@st.cache_data
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


@st.cache_data
def top_results_df(simulated_results, top_n):
    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

# Função para calcular o tamanho da stake com base na responsabilidade desejada


@st.cache_data
def calcular_tamanho_stake(responsabilidade, odd_justa, odd_mercado):
    return responsabilidade / (odd_mercado - 1)

# Função para calcular o tamanho da stake com base na fração de Kelly selecionada


@st.cache_data
def calcular_tamanho_stake_kelly(probabilidade, odd_justa, odd_mercado, kelly_fracao):
    if kelly_fracao == "Kelly Completo":
        return (probabilidade * odd_justa - 1) / (odd_mercado - 1)
    elif kelly_fracao == "Meio Kelly":
        return 0.5 * (probabilidade * odd_justa - 1) / (odd_mercado - 1)
    elif kelly_fracao == "Um Quarto de Kelly":
        return 0.25 * (probabilidade * odd_justa - 1) / (odd_mercado - 1)

# Função para carregar dados de uma URL e tratar o erro 404


def carregar_dados(url):
    try:
        dados = pd.read_csv(url)
        return dados
    except HTTPError as e:
        if e.code == 404:
            st.error('Erro 404: Dados não encontrados:')
        else:
            st.error(f'Erro ao carregar dados: {e}')
        return None
    except Exception as e:
        st.error(f'Erro inesperado: {e}')
        return None


# Função para Configurar a pagina
st.set_page_config(
    page_title="CS Statistic",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titulo da Pagina no sidebar
st.sidebar.title('CS Statistic')

# Escolha do dia por defult retorna o dia atual
dia = st.sidebar.date_input(
    "Selecione a data:", value=datetime.date.today(), key='date_input')
st.markdown("""<style>
    div[data-testid="stDateInput"] > div:first-child {
        width: 50px;
    }
</style>""", unsafe_allow_html=True)

# Titulo da Pagina
st.title('CS Statistic')

# Dicionrio das Ligas
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

# URL do arquivo CSV que você deseja carregar
url = (
    f'https://github.com/futpythontrader/YouTube/blob/main/Jogos_do_Dia/FlashScore/Jogos_do_Dia_FlashScore_{dia}.csv?raw=true')

# Carregar os dados
jogos_do_dia = carregar_dados(url)

# Verificar se os dados foram carregados com sucesso
if jogos_do_dia is not None:
    jogos_do_dia = jogos_do_dia[['League', 'Date',
                                'Time', 'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']]
    jogos_do_dia.columns = ['League', 'Date', 'Time',
                            'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']
    Jogos_do_Dia = jogos_do_dia[jogos_do_dia['League'].isin(leagues) == True]
    Jogos_do_Dia = drop_reset_index(Jogos_do_Dia)

    Jogos = Jogos_do_Dia.sort_values(by='League')
    ligas = Jogos['League'].unique()

    # Importando a Base de Dados
    base = pd.read_csv(
        "https://github.com/futpythontrader/YouTube/blob/main/Bases_de_Dados/FlashScore/Base_de_Dados_FlashScore_v2.csv?raw=true")
    base = base[['League', 'Date', 'Home', 'Away',
                'Goals_H', 'Goals_A', 'Odd_H', 'Odd_D', 'Odd_A']]
    base.columns = ['League', 'Date', 'Home', 'Away',
                    'Goals_H', 'Goals_A', 'Odd_H', 'Odd_D', 'Odd_A']

    base = base[base['League'].isin(ligas) == True]
    base = drop_reset_index(base)

    # Definindo a média do Home e Away nos ultimos 5 Jogos
    n = 5

    base['Media_GM_H'] = base.groupby('Home')['Goals_H'].rolling(
        window=n, min_periods=n).mean().reset_index(0, drop=True)
    base['Media_GM_A'] = base.groupby('Away')['Goals_A'].rolling(
        window=n, min_periods=n).mean().reset_index(0, drop=True)
    base['Media_GM_H'] = base.groupby('Home')['Media_GM_H'].shift(1)
    base['Media_GM_A'] = base.groupby('Away')['Media_GM_A'].shift(1)
    base = drop_reset_index(base)

    base['Media_GS_H'] = base.groupby('Home')['Goals_A'].rolling(
        window=n, min_periods=n).mean().reset_index(0, drop=True)
    base['Media_GS_A'] = base.groupby('Away')['Goals_H'].rolling(
        window=n, min_periods=n).mean().reset_index(0, drop=True)
    base['Media_GS_H'] = base.groupby('Home')['Media_GS_H'].shift(1)
    base['Media_GS_A'] = base.groupby('Away')['Media_GS_A'].shift(1)
    base = drop_reset_index(base)

    # Definindo as medias
    base_H = base[['Home', 'Media_GM_H', 'Media_GS_H']]
    base_A = base[['Away', 'Media_GM_A', 'Media_GS_A']]

    # Criando a Lista com os Confrontos
    Jogos_do_Dia['Jogo'] = Jogos_do_Dia['Home'] + ' x ' + Jogos_do_Dia['Away']
    lista_confrontos = Jogos_do_Dia['Jogo'].tolist()

    # Variavel vazia para armazenar o jogo selecionado
    Jogo = None

    # Função para filtrar e exibir os dados
    def func(jogo_selecionado):
        global Jogo

    # Seleção do Horario
    wid_filtro = st.sidebar.selectbox(
        'Selecione o Horario:', Jogos_do_Dia['Time'].unique(), index=0)
    df_Hora = Jogos_do_Dia[Jogos_do_Dia['Time'] == wid_filtro]
    df_Hora = df_Hora[['League', 'Date', 'Time', 'Home',
                       'Away', 'Odd_H', 'Odd_D', 'Odd_A', 'Jogo']]

    # Seleção da Liga com base o horario
    wid_filtro = st.sidebar.selectbox(
        'Selecione a Liga:', df_Hora['League'].unique(), index=0)
    df_Liga = df_Hora[df_Hora['League'] == wid_filtro]
    df_Liga = df_Liga[['League', 'Date', 'Time', 'Home',
                       'Away', 'Odd_H', 'Odd_D', 'Odd_A', 'Jogo']]

    # Seleção do jogo com base na liga
    wid_filtro = st.sidebar.selectbox(
        'Escolha o Jogo:', df_Liga['Jogo'].unique(), index=0)
    df_filtrado = df_Liga[df_Liga['Jogo'] == wid_filtro]
    df_filtrado = df_filtrado[['League', 'Date', 'Time',
                               'Home', 'Away', 'Odd_H', 'Odd_D', 'Odd_A']]

    # Resentando o index do filtro
    df_filtrado = drop_reset_index(df_filtrado)

    # Exibição do jogo escolhido
    st.write('**Jogo Selecionado:**')
    st.write(df_filtrado)

    try:
        # Variavél recebe o jogo escolhido
        Jogo = df_filtrado

        # Variavel para receber o index do jogo escolhido
        i = 1

        # criando uma base temporaria para receber as medias
        last_base_H = base_H.groupby('Home').last().reset_index()
        last_base_A = base_A.groupby('Away').last().reset_index()

        # Juntando o df como a variavel jogo e last_base
        df = pd.merge(Jogo, last_base_H, how='left',
                      left_on='Home', right_on='Home')
        df = pd.merge(df, last_base_A, how='left',
                      left_on='Away', right_on='Away')
        df = drop_reset_index(df)

        # Recebendo os dados para prepara a simulação de resultado
        Liga = df.loc[i]['League']
        Team_01 = df.loc[i]['Home']
        Team_02 = df.loc[i]['Away']
        Time = df.loc[i]['Time']
        Date = df.loc[i]['Date']

        Media_GM_H = df.loc[i]['Media_GM_H']
        Media_GM_A = df.loc[i]['Media_GM_A']

        Media_GS_H = df.loc[i]['Media_GS_H']
        Media_GS_A = df.loc[i]['Media_GS_A']

        # Simulando o resultado através da função simulate_match
        simulated_results = simulate_match(
            Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
        simulated_results = drop_reset_index(simulated_results)
        results = top_results_df(simulated_results, 10000)
        results = drop_reset_index(results)

        # Fazendo a simulação de placares
        results['Placar'] = results.apply(
            lambda row: 'Goleada_H' if (row['Home_Goals'] >= 4 and row['Home_Goals'] > row['Away_Goals'])
            else 'Goleada_A' if (row['Away_Goals'] >= 4 and row['Away_Goals'] > row['Home_Goals'])
            else 'Goleada_D' if (row['Home_Goals'] >= 4 and row['Away_Goals'] >= 4 and row['Home_Goals'] == row['Away_Goals'])
            else f"{int(row['Home_Goals'])}x{int(row['Away_Goals'])}", axis=1
        )

        # Defina a probabilidade maxima desejada filtrado e arredondando para 2 casas decimais
        probabilidade_maxima = 0.08
        results_filtrado = results[results['Probability']
                                   < probabilidade_maxima].round(2)

        # Calcular as odds justas para os placares selecionados e arredondando a odd justa para duas casas decimais
        results_filtrado['Odd_Justa'] = 1 / \
            results_filtrado['Probability'].round(2)

        # Filtrando os 5 primeiros resultados e resetando o index
        results_filtrado = results_filtrado.head(5)
        results_filtrado = results_filtrado.reset_index(drop=True)
        results_filtrado.index += 1

        # Verifica se base e df_filtrado não estão vazios para mostrar os ultimos 5 jogos home e away
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

        # Classificar em ordem decrescente os ultimos 5 jogos home e away
        last_home_games = last_home_games.sort_values(
            by='Date', ascending=False).reset_index(drop=True)
        last_home_games.index += 1

        last_away_games = last_away_games.sort_values(
            by='Date', ascending=False).reset_index(drop=True)
        last_away_games.index += 1

        # Exibindo os ultimos 5 jogos home e away
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

        # Lista para armazenar as odds de mercado
        odds_mercado = []

        # Adição das odds de mercado ao novo_df
        odds_mercado = []
        for index, row in novo_df.iterrows():
            odd_mercado = st.sidebar.number_input(
                f"Odd Lay '{row['Placar']}':", min_value=1.01, step=0.01)
            odds_mercado.append(odd_mercado)

        # Adição das odds de mercado ao novo_df
        novo_df['Odd_Lay'] = odds_mercado

        # Aplicação do cálculo do tamanho da stake
        novo_df['Stake'] = calcular_tamanho_stake(
            responsabilidade_desejada, novo_df['Odd_Justa'], novo_df['Odd_Lay'])

        # Cálculo do lucro potencial
        comissao = 5.6 / 100  # Comissão de 5.6%
        novo_df['Lucro_Potencial'] = novo_df['Stake'] * (1 - comissao)

        # Calcular o EV para cada placar comparando a odd de mercado com a odd justa
        EV = []
        for index, row in novo_df.iterrows():
            ev_placar = (1 - comissao) * (row['Odd_Justa'] - 1) - \
                (1 - comissao) * (row['Odd_Lay'] - 1)
            EV.append(ev_placar)
        novo_df['EV'] = EV

        # Primeiro código: Arredondar valores e calcular EV
        novo_df['Odd_Justa'] = novo_df['Odd_Justa'].round(2)

        novo_df['EV'] = novo_df['EV'].round(2)
        novo_df['EV+'] = novo_df['EV'] > 0
        novo_df['Stake'] = novo_df['Stake'].round(2)
        novo_df['Lucro_Potencial'] = novo_df['Lucro_Potencial'].round(2)

        # Filtrar os resultados onde o EV é positivo
        results_filtrado_ev_positivo = novo_df[novo_df['EV+']]

        # Aplicar o cálculo do tamanho da stake com base na fração de Kelly selecionada
        novo_df['Porc_Kelly'] = novo_df.apply(lambda row: calcular_tamanho_stake_kelly(
            row['Probability'], row['Odd_Justa'], row['Odd_Lay'], kelly_fracao), axis=1)

        # Arredondar e formatar valores
        novo_df['Porc_Kelly'] = (novo_df['Porc_Kelly'] * 100).round(4)
        novo_df['Porc_Kelly'] = novo_df['Porc_Kelly'] * 100
        novo_df['Porc_Kelly'] = novo_df['Porc_Kelly'].apply(
            lambda x: '{:.2f}%'.format(x))

        # Verificar se a aposta tem EV+ usando o critério de Kelly
        novo_df['EV_Kelly'] = novo_df['Porc_Kelly'].str.rstrip(
            '%').astype(float) > 0

        # Filtrar onde EV+ e EV_Kelly são True
        filtered_df = novo_df[(novo_df['EV+'] == True) &
                              (novo_df['EV_Kelly'] == True)]

        # Resetar o índice do DataFrame filtrado
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df.index = filtered_df.index + 1

        # Cálculo do EV médio, garantindo que há registros no DataFrame filtrado
        if not filtered_df.empty:
            EV_medio = filtered_df['EV'].mean()
        else:
            EV_medio = None

        # Exibição do resultado final
        st.write('Resultados Final:')
        st.write(filtered_df[['Placar', 'Count', 'Probability', 'Odd_Justa',
                              'Odd_Lay', 'Stake', 'Lucro_Potencial', 'EV', 'EV+', 'Porc_Kelly', 'EV_Kelly']])

        # Exibição do cálculo do EV médio, se aplicável
        if EV_medio is not None:
            st.write(f'EV médio: {round(EV_medio, 2)}')
        else:
            st.write('EV médio: Não aplicável')

    except KeyError as e:
        st.write(
            "Erro: Dados não encontrado no DataFrame! Por favor selecione outro jogo.")

else:
    st.write(f'Não há dados disponíveis para o dia {dia}.')
