# %%
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')
# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

# Modelado
# ==============================================================================
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import multiprocessing


# ==============================================================================
df_consumos = pd.read_excel('Consumos_CSV.xlsx')  


df_consumos['Fecha'] = pd.to_datetime(df_consumos['Fecha'], format='%Y-%m-%dT%H:%M:%SZ')
df_consumos = df_consumos.set_index('Fecha')

fin_train = '2019-12-31 23:00:00'
df_consumos_train = df_consumos.loc[: fin_train,:]

print(f"Fechas train      : {df_consumos_train.index.min()} --- {df_consumos_train.index.max()}  (n={len(df_consumos_train)})")



# Gráfico serie temporal
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4))
df_consumos_train['Consumo Total (kWh)'].plot(ax=ax, label='entrenamiento', linewidth=1)
ax.set_title('Demanda de energía')
ax.legend();


# Gráfico boxplot para estacionalidad anual
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
df_consumos['mes'] = df_consumos.index.month
df_consumos.boxplot(column='Consumo Total (kWh)', by='mes', ax=ax,)
df_consumos.groupby('mes')['Consumo Total (kWh)'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Consumo Total (kWh)')
ax.set_title('Distribución demanda por mes')
fig.suptitle('');



# Crear y entrenar forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor     = Ridge(),
                lags          = 24,
                transformer_y = StandardScaler()
             )

forecaster.fit(y=df_consumos.loc[:fin_train, 'Consumo Total (kWh)'])
forecaster


# Backtest
# ==============================================================================
metrica, predicciones = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = df_consumos['Consumo Total (kWh)'],
                            initial_train_size = len(df_consumos.loc[:fin_train])-744,
                            fixed_train_size   = False,
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = True
                        )

df_consumos = df_consumos.reset_index()
prueba2 = df_consumos.iloc[8016:8761,0]
predicciones = predicciones.set_index(prueba2)


# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 3.5))
df_consumos.loc[predicciones.index, 'Consumo Total (kWh)'].plot(ax=ax, linewidth=2, label='test')
predicciones.plot(linewidth=2, label='predicción', ax=ax)
ax.set_title('Predicción vs demanda real')
ax.legend();

# ==============================================================================
# ==============================================================================
# ==============================================================================

df_consumos_enero = df_consumos.iloc[0:744,0]
df_consumos_enero = df_consumos_enero.reset_index()
df_consumos_enero['Fecha'] = pd.to_datetime(df_consumos_enero['Fecha'], format='%Y-%m-%dT%H:%M:%SZ')
df_consumos_enero = df_consumos_enero.set_index('Fecha')
fin_train_enero = '2019-01-24 23:00:00'
df_consumos_enero_train = df_consumos_enero.loc[: fin_train_enero]
df_consumos_enero = df_consumos_enero.to_frame()
df_consumos_enero_train = df_consumos_enero_train.to_frame()

# Gráfico serie temporal mensual
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4))
df_consumos_enero_train['Consumo Total (kWh)'].plot(ax=ax, label='entrenamiento', linewidth=1)
ax.set_title('Demanda de energía mensual')
ax.legend();


# Gráfico boxplot para estacionalidad mensual
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
df_consumos_enero['mes'] = df_consumos_enero.index.hour
df_consumos_enero.boxplot(column='Consumo Total (kWh)', by='mes', ax=ax,)
df_consumos_enero.groupby('mes')['Consumo Total (kWh)'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Consumo Total (kWh)')
ax.set_title('Distribución demanda por mes')
fig.suptitle('');


# Crear y entrenar forecaster mensual
# ==============================================================================
forecaster_mensual = ForecasterAutoreg(
                regressor     = Ridge(),
                lags          = 24,
                transformer_y = StandardScaler()
             )

forecaster_mensual.fit(y=df_consumos_enero.loc[:fin_train_enero, 'Consumo Total (kWh)'])
forecaster_mensual


# Backtest mensual
# ==============================================================================
metrica_mensual, predicciones_mensual = backtesting_forecaster(
                            forecaster = forecaster_mensual,
                            y          = df_consumos_enero['Consumo Total (kWh)'],
                            initial_train_size = len(df_consumos_enero_train),
                            fixed_train_size   = False,
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = True
                        )


df_consumos_enero = df_consumos_enero.reset_index()
prueba2 = df_consumos_enero.iloc[576:744,0]
predicciones_mensual = predicciones_mensual.set_index(prueba2)

# Gráfico mensual
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 3.5))
df_consumos_enero.plot(ax=ax, linewidth=2, label='test')
predicciones_mensual.plot(linewidth=2, label='predicción', ax=ax)
ax.set_title('Predicción vs demanda real')
ax.legend();


forecaster_mensual.predict('2019-01-24 23:00:00')


# ==============================================================================
# ==============================================================================
# ==============================================================================

# ================================ RED NEURONAL ================================

X_train, X_test, y_train, y_test = train_test_split(
                                        df_consumos.drop('Consumo Total (kWh)', axis = 'columns'),
                                        df_consumos['Consumo Total (kWh)'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

print("Partición de entrenamento")
print("-----------------------")
display(y_train.describe())
display(X_train.describe())
display(X_train.describe(include = 'object'))
print(" ")

print("Partición de test")
print("-----------------------")
display(y_test.describe())
display(X_test.describe())
display(X_test.describe(include = 'object'))


categorical_transformer = Pipeline(
                            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
                          )


preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', categorical_transformer, cat_cols)
                    ],
                    remainder='passthrough'
                )


# Se combinan los pasos de preprocesado y el modelo en un mismo pipeline
pipe = Pipeline([('preprocessing', preprocessor),
                 ('modelo', MLPRegressor(solver = 'lbfgs', max_iter= 1000))])

# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
param_distributions = {
    'modelo__hidden_layer_sizes': [(10), (20), (10, 10)],
    'modelo__alpha': np.logspace(-3, 3, 10),
    'modelo__learning_rate_init': [0.001, 0.01],
}

# Búsqueda por validación cruzada
# ==============================================================================
grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 50,
        scoring    = 'neg_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = 5, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

print("New change")


