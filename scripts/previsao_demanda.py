# -*- coding: utf-8 -*-
"""
Projeto de Previsão de Demanda

Este módulo implementa um pipeline completo de previsão de demanda,
incluindo coleta, tratamento, análise exploratória e modelagem preditiva.
"""

import pandas as pd
import numpy as np
import plotnine as p9
from statsmodels.tsa.seasonal import STL
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

from config import (
    DATA_URL, DATE_FORMAT, START_DATE, END_DATE, 
    TRAIN_TEST_SPLIT_DATE, SEASONAL_PERIODS, 
    FORECAST_PERIODS, FIGURE_SIZE, FLOAT_FORMAT
)

# Configurações
pd.options.display.float_format = FLOAT_FORMAT.format

def carregar_dados(url: str) -> pd.DataFrame:
    """
    Carrega dados de demanda de produtos do URL especificado.
    
    Args:
        url: URL do dataset comprimido
        
    Returns:
        DataFrame com dados brutos
    """
    return pd.read_csv(url, compression="zip")


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica tratamentos nos dados brutos.
    
    Args:
        df: DataFrame com dados brutos
        
    Returns:
        DataFrame com dados tratados
    """
    return (
        df.copy()
        .assign(
            Date=lambda x: pd.to_datetime(x["Date"], format=DATE_FORMAT),
            Order_Demand=lambda x: x.Order_Demand.str.replace(
                r"[()]", "", regex=True
            ).astype(int)
        )
    )


def analisar_dados(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza análise exploratória dos dados.
    
    Args:
        df: DataFrame com dados tratados
        
    Returns:
        Dicionário com estatísticas descritivas
    """
    stats = {
        'n_produtos': df.Product_Code.nunique(),
        'n_categorias': df.Product_Category.nunique(),
        'descricao': df.describe(include="all"),
        'categoria_principal': df.Product_Category.value_counts().index[0]
    }
    
    print(f"Número de produtos únicos: {stats['n_produtos']}")
    print(f"Número de categorias: {stats['n_categorias']}")
    print(f"Categoria principal: {stats['categoria_principal']}")
    
    return stats


def preparar_serie_temporal(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    """
    Prepara série temporal para a categoria especificada.
    
    Args:
        df: DataFrame com dados tratados
        categoria: Categoria de produto para análise
        
    Returns:
        DataFrame com série temporal agregada
    """
    return (
        df.query("Product_Category == @categoria")
        .groupby("Date")
        .Order_Demand.sum()
        .to_frame()
        .query("Date >= @pd.to_datetime(@START_DATE)")
        .Order_Demand.resample("W")
        .sum()
        .to_frame()
        .query("index <= @pd.to_datetime(@END_DATE)")
    )


def plotar_serie_temporal(df: pd.DataFrame, titulo: str = "Demanda ao Longo do Tempo"):
    """
    Cria gráfico da série temporal.
    
    Args:
        df: DataFrame com série temporal
        titulo: Título do gráfico
    """
    return (
        p9.ggplot(df.reset_index()) +
        p9.aes(x="Date", y="Order_Demand") +
        p9.geom_line() +
        p9.labs(title=titulo, x="Data", y="Demanda")
    )


def analisar_sazonalidade(serie: pd.Series, periodo: int = 52):
    """
    Realiza decomposição STL da série temporal.
    
    Args:
        serie: Série temporal
        periodo: Período sazonal (52 para semanal anual)
        
    Returns:
        Objeto STL fitted
    """
    stl = STL(serie, period=periodo).fit()
    stl.plot()
    plt.show()
    return stl


def criar_regressores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis regressoras para modelagem.
    
    Args:
        df: DataFrame com série temporal
        
    Returns:
        DataFrame com regressores adicionados
    """
    month = df.index.month
    return df.copy().assign(
        tendencia=lambda x: (df.reset_index().index + 1) + df.Order_Demand.mean(),
        sazonalidade=np.sin(2 * np.pi * month / 12)
    )


def dividir_treino_teste(df: pd.DataFrame, data_corte: str = TRAIN_TEST_SPLIT_DATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide dados em treino e teste.
    
    Args:
        df: DataFrame com dados completos
        data_corte: Data para divisão treino/teste
        
    Returns:
        Tupla com DataFrames de treino e teste
    """
    df_treino = df.query("index <= @pd.to_datetime(@data_corte)").copy()
    df_teste = df.query("index > @pd.to_datetime(@data_corte)").copy()
    return df_treino, df_teste


def treinar_modelo_regressao(df_treino: pd.DataFrame) -> Pipeline:
    """
    Treina modelo de regressão linear.
    
    Args:
        df_treino: DataFrame de treino
        
    Returns:
        Modelo treinado
    """
    modelo = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    
    X = df_treino[["tendencia", "sazonalidade"]]
    y = df_treino["Order_Demand"]
    
    modelo.fit(X, y)
    return modelo


def treinar_modelo_ets(df_treino: pd.DataFrame, periodos_sazonais: int = SEASONAL_PERIODS) -> ExponentialSmoothing:
    """
    Treina modelo ETS (Exponential Smoothing).
    
    Args:
        df_treino: DataFrame de treino
        periodos_sazonais: Número de períodos sazonais
        
    Returns:
        Modelo ETS treinado
    """
    modelo = ExponentialSmoothing(
        df_treino.Order_Demand,
        trend="add",
        seasonal="add",
        seasonal_periods=periodos_sazonais
    ).fit(optimized=True)
    
    return modelo


def fazer_previsoes(modelo_rl: Pipeline, modelo_ets: ExponentialSmoothing, 
                   df_teste: pd.DataFrame) -> pd.DataFrame:
    """
    Gera previsões com ambos os modelos.
    
    Args:
        modelo_rl: Modelo de regressão linear
        modelo_ets: Modelo ETS
        df_teste: DataFrame de teste
        
    Returns:
        DataFrame com previsões e valores reais
    """
    y_prev_rl = modelo_rl.predict(df_teste[["tendencia", "sazonalidade"]])
    y_prev_ets = modelo_ets.forecast(len(df_teste))
    
    return (
        pd.Series(y_prev_rl, index=df_teste.index)
        .rename("RL")
        .to_frame()
        .join(y_prev_ets.rename("ETS"))
        .join(df_teste["Order_Demand"])
    )


def calcular_metricas(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas de erro de previsão.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Dicionário com métricas
    """
    return {
        'erro_medio': np.mean(y_true - y_pred),
        'erro_medio_absoluto': np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(np.mean((y_true - y_pred)**2))
    }


def avaliar_modelos(df_previsao: pd.DataFrame):
    """
    Avalia performance dos modelos e exibe métricas.
    
    Args:
        df_previsao: DataFrame com previsões e valores reais
    """
    y_true = df_previsao.Order_Demand
    
    # Métricas Regressão Linear
    metricas_rl = calcular_metricas(y_true, df_previsao.RL)
    print("=== Métricas Regressão Linear ===")
    print(f"Erro médio: {metricas_rl['erro_medio']:.2f}")
    print(f"Erro médio absoluto: {metricas_rl['erro_medio_absoluto']:.2f}")
    print(f"RMSE: {metricas_rl['rmse']:.2f}\n")
    
    # Métricas ETS
    metricas_ets = calcular_metricas(y_true, df_previsao.ETS)
    print("=== Métricas ETS ===")
    print(f"Erro médio: {metricas_ets['erro_medio']:.2f}")
    print(f"Erro médio absoluto: {metricas_ets['erro_medio_absoluto']:.2f}")
    print(f"RMSE: {metricas_ets['rmse']:.2f}\n")
    
    # Gráfico comparativo
    df_previsao.plot(figsize=FIGURE_SIZE)
    plt.title("Comparação de Previsões")
    plt.ylabel("Demanda")
    plt.show()


def previsao_fora_amostra(df_completo: pd.DataFrame, periodos: int = FORECAST_PERIODS) -> pd.Series:
    """
    Gera previsão fora da amostra usando modelo ETS.
    
    Args:
        df_completo: DataFrame com toda a série temporal
        periodos: Número de períodos para prever
        
    Returns:
        Série com previsões
    """
    modelo_final = ExponentialSmoothing(
        df_completo.Order_Demand,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS
    ).fit(optimized=True)
    
    return modelo_final.forecast(periodos)


def main():
    """
    Função principal que executa todo o pipeline de previsão.
    """
    print("=== Iniciando Pipeline de Previsão de Demanda ===\n")
    
    # 1. Coleta de dados
    print("1. Carregando dados...")
    df_bruto = carregar_dados(DATA_URL)
    
    # 2. Tratamento de dados
    print("2. Tratando dados...")
    df_tratado = tratar_dados(df_bruto)
    
    # 3. Análise exploratória
    print("3. Realizando análise exploratória...")
    stats = analisar_dados(df_tratado)
    
    # 4. Preparação da série temporal
    print("4. Preparando série temporal...")
    categoria_alvo = stats['categoria_principal']
    df_alvo = preparar_serie_temporal(df_tratado, categoria_alvo)
    
    print(f"Série temporal preparada com {len(df_alvo)} observações")
    print(f"Período: {df_alvo.index.min()} a {df_alvo.index.max()}\n")
    
    # 5. Análise de sazonalidade
    print("5. Analisando sazonalidade...")
    stl = analisar_sazonalidade(df_alvo.Order_Demand)
    
    # 6. Preparação para modelagem
    print("6. Preparando dados para modelagem...")
    df_regressao = criar_regressores(df_alvo)
    df_treino, df_teste = dividir_treino_teste(df_regressao)
    
    print(f"Dados de treino: {len(df_treino)} observações")
    print(f"Dados de teste: {len(df_teste)} observações\n")
    
    # 7. Treinamento dos modelos
    print("7. Treinando modelos...")
    modelo_rl = treinar_modelo_regressao(df_treino)
    modelo_ets = treinar_modelo_ets(df_treino)
    
    # 8. Previsões
    print("8. Gerando previsões...")
    df_previsao = fazer_previsoes(modelo_rl, modelo_ets, df_teste)
    
    # 9. Avaliação
    print("9. Avaliando modelos...")
    avaliar_modelos(df_previsao)
    
    # 10. Previsão fora da amostra
    print("10. Gerando previsão fora da amostra...")
    previsao_futura = previsao_fora_amostra(df_alvo)
    
    # Gráfico final
    df_final = df_alvo.join(previsao_futura.rename("ETS_Futuro"), how="outer")
    df_final[["Order_Demand", "ETS_Futuro"]].plot(figsize=FIGURE_SIZE)
    plt.title("Previsão de Demanda - Série Histórica e Projeção")
    plt.ylabel("Demanda")
    plt.show()
    
    print("=== Pipeline concluído com sucesso! ===")


if __name__ == "__main__":
    main()