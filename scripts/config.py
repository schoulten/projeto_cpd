# -*- coding: utf-8 -*-
"""
Configurações do projeto de previsão de demanda.
"""

# URLs e fontes de dados
DATA_URL = "https://aluno.analisemacro.com.br/download/69280/?tmstv=1768230842"

# Parâmetros de processamento
DATE_FORMAT = "%Y/%m/%d"
START_DATE = "2012-01-01"
END_DATE = "2017-01-01"
TRAIN_TEST_SPLIT_DATE = "2015-01-01"

# Parâmetros de modelagem
SEASONAL_PERIODS = 52  # Semanal anual
FORECAST_PERIODS = 52  # Períodos para previsão fora da amostra

# Configurações de visualização
FIGURE_SIZE = (12, 6)
FLOAT_FORMAT = '{:.2f}'