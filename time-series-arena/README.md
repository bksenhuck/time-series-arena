# Time Series Arena

Projeto para previsao de series temporais de visualizacoes de paginas, com backend em FastAPI e frontend em Dash.

## Objetivo

Comparar modelos de forecasting em diferentes horizontes e cenarios de teste, com artefatos precomputados para melhor desempenho.

## Stack

- Python 3.11
- FastAPI
- Dash
- pandas
- xgboost
- statsmodels
- prophet
- pmdarima

## Estrutura principal

- app/: frontend Dash e callbacks
- backend/: API e servicos
- ml/: pipeline, modelos e metricas
- config/: configuracoes e caminhos
- data/: base raw e artefatos processados

## Como executar localmente

1. Criar e ativar ambiente virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Executar a aplicacao:

```bash
python main.py
```

## Pipeline de precomputacao

Para gerar artefatos de treino, teste e comparacao de cenarios:

```bash
python -c "from ml.pipeline.training_pipeline import run_pipeline; print(run_pipeline(force_retrain=False))"
```

## API principal

- GET /api/health
- GET /api/series/{page}
- GET /api/forecast?page={page}&horizon={dias}
- GET /api/metrics?page={page}
- GET /api/scenarios/compare?page={page}

## Observacoes

- Os artefatos processados sao gravados em Parquet.
- Cenarios padrao de comparacao: 3m, 6m e 9m.
