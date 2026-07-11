# 📊 Plataforma Avanzada de People Analytics & Business Intelligence

## 👥 Autores

Álex Hevia
Yurfran Uzcategui
Milton Rojas

---

## 📝 Descripción del Proyecto
Este ecosistema modular de **Business Intelligence (BI)** y **People Analytics** está diseñado para el diagnóstico, análisis multivariante y mitigación de la deserción laboral (fuga de talentos) dentro de una organización. El sistema consolida múltiples fuentes de datos distribuidas (clínicas, demográficas y de gestión corporativa) y aplica modelos avanzados de Inteligencia Artificial para predecir el riesgo de abandono voluntario, evaluando en tiempo real el Retorno de Inversión (ROI) de las medidas correctivas.

La arquitectura de la solución se basa en microservicios independientes que interactúan de forma nativa para la orquestación y flujo de la información.

---

## 🏛️ Arquitectura Tecnológica y Componentes

La solución se divide en tres capas principales integradas de forma nativa:

1. **Capa de Almacenamiento (Persistencia):** Base de datos relacional robusta en **PostgreSQL**, encargada de almacenar las líneas base corporativas, registros históricos de ausentismo y KPIs operativos, corriendo de forma aislada dentro de un contenedor Docker.
2. **Capa de Negocio (Back-End API):** Microservicio desarrollado en **FastAPI (Python)**. Extrae la información de la base de datos, aplica lógica de negocio y expone endpoints RESTful optimizados y protegidos para el consumo de datos en tiempo real mediante paginación dinámica.
3. **Capa de Presentación (Front-End Dashboard):** Interfaz analítica interactiva desarrollada en **Streamlit**. Se conecta dinámicamente al Back-End y despliega tableros multivariantes, simuladores predictivos, métricas financieras de ROI y visualizaciones avanzadas utilizando **Plotly**.

Todo el ecosistema se encuentra orquestado mediante **Docker** y **Docker Compose**.

---

## 🔮 Inteligencia Artificial e Ingeniería de Datos

La suite predictiva integra dos enfoques de machine learning utilizando la librería especializada `scikit-learn`:

### 1. Modelo Predictivo Supervisado: Random Forest (Bosque Aleatorio)
* **Función:** Consume variables analíticas (ausentismo, rendimiento, riesgo de burnout, nivel de estrés, calidad del sueño y soporte gerencial) para calcular la probabilidad exacta ($0\%$ a $100\%$) de deserción laboral de un colaborador.
* **Semáforo Predictivo (Tricolor):** Clasifica el riesgo de forma dinámica en tres umbrales de alerta temprana cambiando los componentes visuales en tiempo real:
  * 🟢 **Riesgo Bajo ($\le 40\%$):** Zona segura. Mantenimiento pasivo del clima.
  * 🟡 **Riesgo Medio ($40\% \text{ a } 65\%$):** Alerta temprana. Sugiere medidas preventivas de bajo costo (ajustes de jornada, feedback 1-a-1, capacitaciones a jefaturas).
  * 🔴 **Riesgo Alto ($> 65\%$):** Alerta crítica. Gatilla planes de contención inmediata (planes de retención salarial, de desarrollo de carrera, auditoría de liderazgo directo, derivación prioritaria al PAE).
* **Feature Importance:** Permite la interpretabilidad del modelo (XAI), exponiendo en un gráfico horizontal el peso analítico real que el algoritmo otorga a cada indicador corporativo en la toma de decisiones.

### 2. Modelo de Clustering No Supervisado: K-Means (K-Medios)
* **Función:** Segmenta la fuerza laboral de manera automatizada en **3 clústeres** analizando la convergencia multivariante de *Ausentismo*, *Burnout* e *Impacto de Productividad*.
* **Resultado:** Clasifica a la dotación en perfiles organizacionales claros (*Estables / Bajo Desgaste*, *Riesgo Latente*, *Desgaste Crítico / Alerta*),iniendo a la alta gerencia aplicar políticas de capital humano masivas y eficientes.

---

## 💰 Módulo Financiero y Análisis de Retorno de Inversión (ROI)

La plataforma cuenta con un motor de Business Intelligence que traduce las métricas de salud laboral e IA en impacto financiero cuantificable:
* **Costo de Deserción (Pérdida):** Calcula el costo hundido que significa dejar ir a un trabajador, indexando la pérdida acumulada de productividad por ausentismo operativo junto al costo de reclutamiento y onboarding de un reemplazo (estimado en el $20\%$ del salario anual corporativo).
* **Costo de Mitigación (Inversión):** Modula un presupuesto institucional fijo para la ejecución de los planes de acción prescriptivos sugeridos por la IA (programas PAE de salud e higiene del sueño, pausas activas, etc.).
* **Ahorro Neto:** Expone visualmente la ganancia financiera y el valor económico retenido dentro de la compañía al mitigar el riesgo preventivamente mediante el uso del software.

---

## 🚀 Guía de Despliegue: Orden de Encendido

Para iniciar correctamente el ecosistema, abre tres consolas (terminales) de forma independiente y sigue estrictamente este orden:

### Paso 1: Levantar la Base de Datos (Contenedor Docker)
Asegúrate de tener la aplicación **Docker Desktop** abierta en segundo plano. Abre tu primera terminal y ejecuta:

cd C:\Users\milto\Desktop\sistema_salud_laboral
docker-compose up -d

###Paso 2: Iniciar el Back-End (FastAPI)

cd C:\Users\milto\Desktop\sistema_salud_laboral
py -m uvicorn api.main:app --reload

###Paso 3: Iniciar el Front-End (Streamlit Dashboard)
cd C:\Users\milto\Desktop\sistema_salud_laboral
py -m streamlit run dashboards/app.py

###Resolución de Problemas Comunes y Reinicios

1. Error de Conexión a la Base de Datos 
# 1. Detiene y destruye los contenedores e instancias activas del proyecto
docker-compose down

# 2. Levanta los servicios desde cero forzando la reconstrucción limpia de la caché
docker-compose up --build -d


📁 Estructura de Carpetas del Repositorio
├── data/
│   └── mental_health_workplace (1).csv  # Dataset base demográfico
├── dashboards/
│   └── app.py                            # Aplicación Streamlit (BI & IA Dashboard)
├── api/
│   └── main.py                           # API RESTful en FastAPI
├── docker-compose.yml                    # Orquestador de contenedores Docker
└── README.md                             # Documentación técnica institucional
