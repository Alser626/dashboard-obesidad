# Dashboard – Análisis de Factores Asociados a la Obesidad

**Autor:** Sergio Roa  
**Curso:** Ingeniería en Ciencia de Datos  
**Año:** 2026

## Descripción

Dashboard interactivo desarrollado con Dash y Plotly que presenta los resultados del análisis exploratorio de datos sobre factores asociados a la obesidad.

## Contenido del Dashboard

- **Exploración:** Histogramas, gráficos de barras y boxplots interactivos
- **Hipótesis:** Contraste de hipótesis (prueba t de Student)
- **Modelos:** Regresión lineal múltiple y regresión logística
- **Predictor:** Herramienta interactiva para estimar el IMC de un paciente
- **Conclusiones:** Hallazgos y recomendaciones clínicas

## Instalación y ejecución local

```bash
# 1. Clonar el repositorio
git clone github.com/Alser626/dashboard-obesidad
cd dashboard-obesidad

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el dashboard
python app.py

# 4. Abrir en el navegador
# https://8050-m-s-948leo2alv1j-d.us-east1-0.prod.colab.dev/```

## Dataset

Dataset sintético basado en: Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition*. Data in Brief, 25, 104344.

## Referencias

- OMS – Obesidad: https://www.who.int/es/news-room/fact-sheets/detail/obesity-and-overweight
- Dash Documentation: https://dash.plotly.com/
- Scikit-learn: https://scikit-learn.org/stable/
