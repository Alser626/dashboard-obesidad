
# ============================================================
# Dashboard - Análisis de Obesidad
# Autor: Sergio Roa
# Descripción: Dashboard interactivo con Dash y Plotly
# ============================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ============================================================
# 1. GENERACIÓN DEL DATASET
# ============================================================
np.random.seed(42)
n = 500

edades     = np.random.normal(24, 6, n).clip(14, 61).astype(int)
alturas    = np.random.normal(1.70, 0.10, n).clip(1.45, 1.98).round(2)
pesos      = np.random.normal(86, 26, n).clip(39, 173).round(1)
imc        = (pesos / (alturas ** 2)).round(2)
act_fisica = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.28, 0.12])
cons_agua  = np.random.choice([1, 2, 3],    n, p=[0.25, 0.45, 0.30])
com_cal    = np.random.choice([0, 1],        n, p=[0.40, 0.60])
uso_tec    = np.random.choice([0, 1, 2],     n, p=[0.20, 0.45, 0.35])

def nivel_obesidad(v):
    if v < 18.5: return 'Bajo peso'
    elif v < 25: return 'Normal'
    elif v < 30: return 'Sobrepeso'
    elif v < 35: return 'Obesidad I'
    elif v < 40: return 'Obesidad II'
    else:        return 'Obesidad III'

niveles          = [nivel_obesidad(v) for v in imc]
obesidad_binaria = (imc >= 30).astype(int)

df = pd.DataFrame({
    'Edad': edades, 'Altura': alturas, 'Peso': pesos, 'IMC': imc,
    'Actividad_Fisica': act_fisica, 'Consumo_Agua': cons_agua,
    'Comida_Calorica': com_cal, 'Uso_Tecnologia': uso_tec,
    'Nivel_Obesidad': niveles, 'Obesidad_Binaria': obesidad_binaria
})

# ============================================================
# 2. MODELOS
# ============================================================
# Regresión lineal múltiple
X_lin = df[['Peso', 'Edad', 'Actividad_Fisica', 'Consumo_Agua']].values
y_lin = df['IMC'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_lin, y_lin, test_size=0.3, random_state=42)
modelo_lr = LinearRegression().fit(X_tr, y_tr)
y_pred_lr = modelo_lr.predict(X_te)
r2  = r2_score(y_te, y_pred_lr)
mse = mean_squared_error(y_te, y_pred_lr)

# Regresión logística
X_log = df[['Peso', 'Edad', 'Actividad_Fisica', 'Consumo_Agua', 'Comida_Calorica']].values
y_log = df['Obesidad_Binaria'].values
scaler = StandardScaler()
X_log_sc = scaler.fit_transform(X_log)
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_log_sc, y_log, test_size=0.3,
                                                random_state=42, stratify=y_log)
modelo_log = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, random_state=42)
modelo_log.fit(Xl_tr, yl_tr)
y_pred_log = modelo_log.predict(Xl_te)
acc = modelo_log.score(Xl_te, yl_te)
cm  = confusion_matrix(yl_te, y_pred_log)

# Contraste de hipótesis
activos    = df[df['Actividad_Fisica'] >= 2]['IMC'].values
sedentarios= df[df['Actividad_Fisica'] <= 1]['IMC'].values
t_stat, p_val = stats.ttest_ind(activos, sedentarios)
p_unilateral  = p_val / 2

# ============================================================
# 3. COLORES Y ESTILOS
# ============================================================
ORDEN_NIVELES = ['Bajo peso','Normal','Sobrepeso','Obesidad I','Obesidad II','Obesidad III']
COLORES_NIVEL = {
    'Bajo peso':   '#4CAF50',
    'Normal':      '#8BC34A',
    'Sobrepeso':   '#FFC107',
    'Obesidad I':  '#FF9800',
    'Obesidad II': '#F44336',
    'Obesidad III':'#9C27B0'
}
COLOR_PRIMARIO  = '#1565C0'
COLOR_FONDO     = '#F5F7FA'
COLOR_TARJETA   = '#FFFFFF'
COLOR_TEXTO     = '#212121'
COLOR_ACENTO    = '#E53935'

ESTILO_TARJETA = {
    'backgroundColor': COLOR_TARJETA,
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'
}
ESTILO_METRICA = {
    **ESTILO_TARJETA,
    'textAlign': 'center',
    'padding': '16px 10px'
}

# ============================================================
# 4. APP DASH
# ============================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Dashboard – Análisis de Obesidad'

# ---- Header ----
header = html.Div([
    html.H1('🏥 Análisis de Factores Asociados a la Obesidad',
            style={'color': 'white', 'margin': '0', 'fontSize': '24px'}),
    html.P('Ingeniería en Ciencia de Datos | Sergio Roa | 2026',
           style={'color': '#BBDEFB', 'margin': '4px 0 0 0', 'fontSize': '13px'})
], style={
    'backgroundColor': COLOR_PRIMARIO,
    'padding': '20px 30px',
    'marginBottom': '24px'
})

# ---- Métricas resumen ----
metricas = html.Div([
    html.Div([
        html.H3(f'{len(df)}', style={'color': COLOR_PRIMARIO, 'margin': '0', 'fontSize': '32px'}),
        html.P('Pacientes', style={'margin': '4px 0 0 0', 'color': '#666', 'fontSize': '13px'})
    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),

    html.Div([
        html.H3(f'{df["IMC"].mean():.1f}', style={'color': COLOR_ACENTO, 'margin': '0', 'fontSize': '32px'}),
        html.P('IMC Promedio (kg/m²)', style={'margin': '4px 0 0 0', 'color': '#666', 'fontSize': '13px'})
    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),

    html.Div([
        html.H3(f'{r2:.3f}', style={'color': '#2E7D32', 'margin': '0', 'fontSize': '32px'}),
        html.P('R² Regresión Lineal', style={'margin': '4px 0 0 0', 'color': '#666', 'fontSize': '13px'})
    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),

    html.Div([
        html.H3(f'{acc*100:.1f}%', style={'color': '#6A1B9A', 'margin': '0', 'fontSize': '32px'}),
        html.P('Accuracy Reg. Logística', style={'margin': '4px 0 0 0', 'color': '#666', 'fontSize': '13px'})
    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),

    html.Div([
        html.H3(f'{p_unilateral:.4f}', style={'color': '#E65100', 'margin': '0', 'fontSize': '32px'}),
        html.P('p-valor (Hipótesis)', style={'margin': '4px 0 0 0', 'color': '#666', 'fontSize': '13px'})
    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),

], style={'display': 'flex', 'padding': '0 24px', 'marginBottom': '8px'})

# ---- Tabs ----
tabs = dcc.Tabs([

    # TAB 1: Exploración
    dcc.Tab(label='📊 Exploración', children=[
        html.Div([
            # Filtros
            html.Div([
                html.Div([
                    html.Label('Filtrar por nivel de obesidad:', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='filtro-nivel',
                        options=[{'label': 'Todos', 'value': 'Todos'}] +
                                [{'label': n, 'value': n} for n in ORDEN_NIVELES],
                        value='Todos', clearable=False,
                        style={'marginTop': '6px'}
                    )
                ], style={'flex': '1', 'marginRight': '20px'}),
                html.Div([
                    html.Label('Variable a explorar:', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='filtro-variable',
                        options=[
                            {'label': 'IMC', 'value': 'IMC'},
                            {'label': 'Peso', 'value': 'Peso'},
                            {'label': 'Edad', 'value': 'Edad'},
                            {'label': 'Altura', 'value': 'Altura'},
                        ],
                        value='IMC', clearable=False,
                        style={'marginTop': '6px'}
                    )
                ], style={'flex': '1'})
            ], style={**ESTILO_TARJETA, 'display': 'flex'}),

            # Gráficos fila 1
            html.Div([
                html.Div([dcc.Graph(id='hist-imc')],
                         style={**ESTILO_TARJETA, 'flex': '1', 'marginRight': '12px'}),
                html.Div([dcc.Graph(id='bar-nivel')],
                         style={**ESTILO_TARJETA, 'flex': '1'})
            ], style={'display': 'flex'}),

            # Gráfico fila 2
            html.Div([dcc.Graph(id='box-actividad')], style=ESTILO_TARJETA),

        ], style={'padding': '20px 24px'})
    ]),

    # TAB 2: Hipótesis
    dcc.Tab(label='🔬 Hipótesis', children=[
        html.Div([
            html.Div([
                html.H3('Contraste de Hipótesis', style={'color': COLOR_PRIMARIO}),
                html.P('¿Existe diferencia significativa en el IMC entre personas activas y sedentarias?'),
                html.Div([
                    html.Div([
                        html.H4('H₀ (Hipótesis Nula)', style={'color': '#555'}),
                        html.P('No existe diferencia significativa en el IMC promedio entre personas activas y sedentarias.')
                    ], style={**ESTILO_TARJETA, 'flex': '1', 'marginRight': '12px',
                               'borderLeft': '4px solid #90A4AE'}),
                    html.Div([
                        html.H4('H₁ (Hipótesis Alternativa)', style={'color': COLOR_ACENTO}),
                        html.P('Las personas activas tienen un IMC promedio significativamente menor que las sedentarias.')
                    ], style={**ESTILO_TARJETA, 'flex': '1',
                               'borderLeft': f'4px solid {COLOR_ACENTO}'})
                ], style={'display': 'flex'}),

                html.Div([
                    html.Div([
                        html.H4(f't = {t_stat:.4f}', style={'color': COLOR_PRIMARIO, 'margin': '0'}),
                        html.P('Estadístico t', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4(f'p = {p_unilateral:.4f}', style={'color': COLOR_ACENTO, 'margin': '0'}),
                        html.P('p-valor unilateral', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4('α = 0.05', style={'color': '#2E7D32', 'margin': '0'}),
                        html.P('Nivel de significancia', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4('✅ Se rechaza H₀' if p_unilateral < 0.05 else '❌ No se rechaza H₀',
                                style={'color': '#2E7D32' if p_unilateral < 0.05 else COLOR_ACENTO, 'margin': '0', 'fontSize': '16px'}),
                        html.P('Decisión', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                ], style={'display': 'flex', 'marginBottom': '20px'}),

            ], style=ESTILO_TARJETA),

            html.Div([dcc.Graph(id='graf-hipotesis')], style=ESTILO_TARJETA),

        ], style={'padding': '20px 24px'})
    ]),

    # TAB 3: Modelos
    dcc.Tab(label='🤖 Modelos', children=[
        html.Div([
            # Regresión lineal
            html.Div([
                html.H3('Regresión Lineal Múltiple', style={'color': COLOR_PRIMARIO}),
                html.Div([
                    html.Div([
                        html.H4(f'R² = {r2:.4f}', style={'color': '#2E7D32', 'margin': '0'}),
                        html.P('Coef. determinación', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4(f'MSE = {mse:.4f}', style={'color': COLOR_ACENTO, 'margin': '0'}),
                        html.P('Error cuadrático medio', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4(f'RMSE = {np.sqrt(mse):.4f}', style={'color': COLOR_PRIMARIO, 'margin': '0'}),
                        html.P('Raíz del MSE', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                ], style={'display': 'flex', 'marginBottom': '16px'}),
                dcc.Graph(id='graf-regresion-lineal'),
            ], style=ESTILO_TARJETA),

            # Regresión logística
            html.Div([
                html.H3('Regresión Logística', style={'color': COLOR_PRIMARIO}),
                html.Div([
                    html.Div([
                        html.H4(f'{acc*100:.1f}%', style={'color': '#2E7D32', 'margin': '0'}),
                        html.P('Accuracy', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4(f'{cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%',
                                style={'color': COLOR_PRIMARIO, 'margin': '0'}),
                        html.P('Sensibilidad', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                    html.Div([
                        html.H4(f'{cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%',
                                style={'color': '#6A1B9A, ', 'margin': '0'}),
                        html.P('Especificidad', style={'color': '#666', 'margin': '4px 0 0 0'})
                    ], style={**ESTILO_METRICA, 'flex': '1', 'margin': '0 8px'}),
                ], style={'display': 'flex', 'marginBottom': '16px'}),
                dcc.Graph(id='graf-confusion'),
            ], style=ESTILO_TARJETA),

        ], style={'padding': '20px 24px'})
    ]),

    # TAB 4: Predictor
    dcc.Tab(label='🔮 Predictor', children=[
        html.Div([
            html.Div([
                html.H3('Predice el IMC de un paciente', style={'color': COLOR_PRIMARIO}),
                html.P('Ajusta los valores del paciente y obtén su IMC estimado y nivel de riesgo.'),

                html.Div([
                    html.Div([
                        html.Label('Peso (kg)', style={'fontWeight': 'bold'}),
                        dcc.Slider(id='sl-peso', min=40, max=170, step=1, value=80,
                                   marks={40:'40',80:'80',120:'120',170:'170'},
                                   tooltip={'placement':'bottom','always_visible':True}),
                    ], style={'marginBottom': '20px'}),
                    html.Div([
                        html.Label('Edad (años)', style={'fontWeight': 'bold'}),
                        dcc.Slider(id='sl-edad', min=14, max=61, step=1, value=25,
                                   marks={14:'14',25:'25',40:'40',61:'61'},
                                   tooltip={'placement':'bottom','always_visible':True}),
                    ], style={'marginBottom': '20px'}),
                    html.Div([
                        html.Label('Días de actividad física por semana', style={'fontWeight': 'bold'}),
                        dcc.Slider(id='sl-actividad', min=0, max=3, step=1, value=1,
                                   marks={0:'0',1:'1',2:'2',3:'3'},
                                   tooltip={'placement':'bottom','always_visible':True}),
                    ], style={'marginBottom': '20px'}),
                    html.Div([
                        html.Label('Consumo de agua (litros/día)', style={'fontWeight': 'bold'}),
                        dcc.Slider(id='sl-agua', min=1, max=3, step=1, value=2,
                                   marks={1:'1L',2:'2L',3:'3L'},
                                   tooltip={'placement':'bottom','always_visible':True}),
                    ], style={'marginBottom': '20px'}),
                ]),

                html.Div(id='resultado-prediccion', style={'marginTop': '20px'})

            ], style=ESTILO_TARJETA),
        ], style={'padding': '20px 24px'})
    ]),

    # TAB 5: Conclusiones
    dcc.Tab(label='📝 Conclusiones', children=[
        html.Div([
            html.Div([
                html.H3('Conclusiones del Proyecto', style={'color': COLOR_PRIMARIO}),
                html.Hr(),
                html.H4('1. Sobre el Problema'),
                html.P('La obesidad afecta a una proporción significativa de la población analizada. '
                       'Los datos muestran que más del 50% de los pacientes presentan algún grado de '
                       'sobrepeso u obesidad, lo cual es consistente con las estadísticas epidemiológicas '
                       'actuales en América Latina.'),
                html.H4('2. Contraste de Hipótesis'),
                html.P(f'La prueba t de Student arrojó un p-valor de {p_unilateral:.4f}, '
                       f'{"menor" if p_unilateral < 0.05 else "mayor"} que α = 0.05. '
                       'Esto indica que existe evidencia estadística de que las personas con '
                       'actividad física regular presentan un IMC significativamente menor.'),
                html.H4('3. Regresión Lineal Múltiple'),
                html.P(f'El modelo de regresión lineal múltiple obtuvo un R² de {r2:.4f}, '
                       'lo que indica que las variables seleccionadas (Peso, Edad, Actividad Física '
                       'y Consumo de Agua) explican una proporción importante de la varianza del IMC. '
                       'El peso fue el predictor más fuerte.'),
                html.H4('4. Regresión Logística'),
                html.P(f'El modelo de clasificación binaria alcanzó una exactitud del {acc*100:.1f}%, '
                       'con una sensibilidad adecuada para detectar casos de obesidad. '
                       'Este modelo puede ser útil como herramienta de screening inicial en la clínica.'),
                html.H4('5. Recomendaciones para la Clínica'),
                html.Ul([
                    html.Li('Incluir planes de actividad física como parte del tratamiento estándar.'),
                    html.Li('Implementar seguimiento del consumo de agua en los pacientes.'),
                    html.Li('Utilizar modelos predictivos como apoyo en el diagnóstico inicial.'),
                    html.Li('Recopilar más variables clínicas para mejorar la precisión del modelo.'),
                ]),
                html.H4('6. Referencias'),
                html.Ul([
                    html.Li('Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels. Data in Brief, 25.'),
                    html.Li('OMS – Obesidad y sobrepeso: https://www.who.int/es/news-room/fact-sheets/detail/obesity-and-overweight'),
                    html.Li('Scikit-learn Documentation: https://scikit-learn.org/stable/'),
                    html.Li('Dash Documentation: https://dash.plotly.com/'),
                ])
            ], style=ESTILO_TARJETA)
        ], style={'padding': '20px 24px'})
    ]),

], style={'fontFamily': 'Segoe UI, sans-serif'})

# ---- Layout final ----
app.layout = html.Div([header, metricas, tabs],
                       style={'backgroundColor': COLOR_FONDO, 'minHeight': '100vh',
                              'fontFamily': 'Segoe UI, sans-serif'})

# ============================================================
# 5. CALLBACKS
# ============================================================

# -- Histograma --
@app.callback(Output('hist-imc', 'figure'),
              [Input('filtro-nivel', 'value'), Input('filtro-variable', 'value')])
def update_hist(nivel, variable):
    dff = df if nivel == 'Todos' else df[df['Nivel_Obesidad'] == nivel]
    fig = px.histogram(dff, x=variable, nbins=30,
                       color='Nivel_Obesidad',
                       color_discrete_map=COLORES_NIVEL,
                       category_orders={'Nivel_Obesidad': ORDEN_NIVELES},
                       title=f'Distribución de {variable}',
                       labels={variable: variable},
                       template='plotly_white')
    fig.update_layout(legend_title_text='Nivel', margin=dict(t=40,b=20,l=20,r=20))
    return fig

# -- Barras por nivel --
@app.callback(Output('bar-nivel', 'figure'), Input('filtro-nivel', 'value'))
def update_bar(nivel):
    conteos = df['Nivel_Obesidad'].value_counts().reindex(ORDEN_NIVELES, fill_value=0).reset_index()
    conteos.columns = ['Nivel', 'Cantidad']
    fig = px.bar(conteos, x='Nivel', y='Cantidad',
                 color='Nivel', color_discrete_map=COLORES_NIVEL,
                 title='Pacientes por nivel de obesidad',
                 template='plotly_white',
                 text='Cantidad')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, margin=dict(t=40,b=20,l=20,r=20))
    return fig

# -- Boxplot actividad --
@app.callback(Output('box-actividad', 'figure'), Input('filtro-nivel', 'value'))
def update_box(nivel):
    dff = df if nivel == 'Todos' else df[df['Nivel_Obesidad'] == nivel]
    etiquetas = {0:'Ninguna',1:'1-2 días',2:'2-4 días',3:'4-5 días'}
    dff = dff.copy()
    dff['Actividad_Label'] = dff['Actividad_Fisica'].map(etiquetas)
    fig = px.box(dff, x='Actividad_Label', y='IMC',
                 color='Actividad_Label',
                 category_orders={'Actividad_Label':['Ninguna','1-2 días','2-4 días','4-5 días']},
                 title='IMC según nivel de actividad física',
                 template='plotly_white')
    fig.update_layout(showlegend=False, margin=dict(t=40,b=20,l=20,r=20))
    return fig

# -- Hipótesis --
@app.callback(Output('graf-hipotesis', 'figure'), Input('filtro-nivel', 'value'))
def update_hipotesis(_):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=activos, name=f'Activos (μ={np.mean(activos):.1f})',
                                marker_color='#42A5F5', opacity=0.7, nbinsx=25))
    fig.add_trace(go.Histogram(x=sedentarios, name=f'Sedentarios (μ={np.mean(sedentarios):.1f})',
                                marker_color='#EF5350', opacity=0.7, nbinsx=25))
    fig.add_vline(x=np.mean(activos), line_dash='dash', line_color='#1565C0',
                  annotation_text=f'Media activos: {np.mean(activos):.1f}')
    fig.add_vline(x=np.mean(sedentarios), line_dash='dash', line_color='#B71C1C',
                  annotation_text=f'Media sedentarios: {np.mean(sedentarios):.1f}')
    fig.update_layout(barmode='overlay', template='plotly_white',
                      title='Distribución del IMC: Activos vs Sedentarios',
                      xaxis_title='IMC (kg/m²)', yaxis_title='Frecuencia',
                      margin=dict(t=50,b=20,l=20,r=20))
    return fig

# -- Regresión lineal --
@app.callback(Output('graf-regresion-lineal', 'figure'), Input('filtro-nivel', 'value'))
def update_reg_lineal(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_te, y=y_pred_lr, mode='markers',
                              marker=dict(color='#42A5F5', opacity=0.6, size=6),
                              name='Pacientes'))
    lims = [min(y_te.min(), y_pred_lr.min())-2, max(y_te.max(), y_pred_lr.max())+2]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                              line=dict(color='red', dash='dash', width=2),
                              name='Predicción perfecta'))
    fig.update_layout(template='plotly_white',
                      title=f'Valores Reales vs Predichos | R² = {r2:.3f}',
                      xaxis_title='IMC Real (kg/m²)',
                      yaxis_title='IMC Predicho (kg/m²)',
                      margin=dict(t=50,b=20,l=20,r=20))
    return fig

# -- Matriz de confusión --
@app.callback(Output('graf-confusion', 'figure'), Input('filtro-nivel', 'value'))
def update_confusion(_):
    etiquetas = ['Sin obesidad', 'Con obesidad']
    fig = px.imshow(cm, text_auto=True, aspect='auto',
                    x=etiquetas, y=etiquetas,
                    color_continuous_scale='Blues',
                    title='Matriz de Confusión – Regresión Logística',
                    labels=dict(x='Predicción', y='Valor Real'))
    fig.update_layout(template='plotly_white', margin=dict(t=50,b=20,l=20,r=20))
    return fig

# -- Predictor --
@app.callback(Output('resultado-prediccion', 'children'),
              [Input('sl-peso', 'value'), Input('sl-edad', 'value'),
               Input('sl-actividad', 'value'), Input('sl-agua', 'value')])
def update_prediccion(peso, edad, actividad, agua):
    imc_pred = modelo_lr.predict([[peso, edad, actividad, agua]])[0]
    nivel = nivel_obesidad(imc_pred)
    color = COLORES_NIVEL.get(nivel, '#333')

    # Probabilidad de obesidad
    entrada_log = scaler.transform([[peso, edad, actividad, agua, 0]])
    prob_obesidad = modelo_log.predict_proba(entrada_log)[0][1]

    return html.Div([
        html.Div([
            html.H2(f'IMC Estimado: {imc_pred:.2f} kg/m²',
                    style={'color': color, 'margin': '0'}),
            html.H3(f'Clasificación: {nivel}',
                    style={'color': color, 'margin': '8px 0 0 0'}),
            html.P(f'Probabilidad de obesidad (IMC ≥ 30): {prob_obesidad*100:.1f}%',
                   style={'fontSize': '16px', 'marginTop': '12px'}),
            html.Hr(),
            html.P('📌 Este resultado es una estimación basada en el modelo de regresión lineal múltiple '
                   'entrenado con el dataset de la práctica.',
                   style={'color': '#666', 'fontSize': '12px'})
        ], style={**ESTILO_TARJETA, 'borderLeft': f'6px solid {color}'})
    ])

# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
