# etl/etl.py
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Agregar la ruta raíz del proyecto para poder importar la API correctamente
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import api.models as models

# --- CONFIGURACIÓN DE LA BASE DE DATOS (DOCKER) ---
DATABASE_URL = "postgresql://postgres:admin123_duoc@localhost:5433/db_salud_laboral"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def conectar_db():
    """Intenta conectar con la base de datos en Docker con manejo de errores avanzado."""
    try:
        db = SessionLocal()
        # Consulta nativa universal que no requiere tablas existentes
        db.execute(text("SELECT 1"))
        print("[ETL] Conexión exitosa a la base de datos en Docker.")
        return db
    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudo conectar a la base de datos en PostgreSQL: {e}")
        print("[CONSEJO] Asegúrate de que el contenedor de Docker esté encendido (docker-compose up -d).")
        sys.exit(1)

def ejecutar_pipeline():
    print("[ETL] Iniciando Pipeline de forma automatizada...")
    
    # RUTA DEL DATASET
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mental_health_workplace (1).csv")
    
    # 1. EXTRACCIÓN
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el dataset maestro en: {csv_path}")
        sys.exit(1)
        
    print(f"[ETL] Leyendo dataset maestro desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. TRANSFORMACIÓN Y LIMPIEZA ROBUSTA
    print("[ETL] Aplicando transformaciones robustas y limpieza...")
    df.columns = df.columns.str.strip() # Eliminar espacios en los nombres de columnas
    
    # Manejo de valores nulos o vacíos en columnas críticas
    df['stress_level'] = df['stress_level'].fillna('Unknown')
    df['company_size'] = df['company_size'].fillna('Unknown')
    df['country'] = df['country'].fillna('Not Specified')
    
    print("[ETL] Simulando extracción y aislamiento de Fuente 1 (Demográficos CSV) y Fuente 2 (Clínicos API)...")
    
    # 3. CARGA EN POSTGRESQL (DOCKER)
    db = conectar_db()
    
    print("[ETL] Limpiando tablas previas mediante recreación del esquema...")
    # Borra y recrea las tablas de forma limpia garantizando la reproducibilidad
    models.Base.metadata.drop_all(bind=engine)
    models.Base.metadata.create_all(bind=engine)
    
    try:
        print("[ETL] Insertando 10000 registros normalizados en PostgreSQL...")
        
        # Iteramos usando estrictamente la variable 'row' de principio a fin
        for index, row in df.iterrows():
            
            # PARTE A: Tabla Base (Fuente 1 + País)
            nuevo_empleado = models.EmpleadoBase(
                record_id=str(row['record_id']),
                company_size=str(row['company_size']),
                country=str(row['country']) # <-- Integrado con éxito
            )
            db.add(nuevo_empleado)
            
            # PARTE B: Condiciones Operativas (Fuente 2)
            nueva_operativa = models.CondicionOperativa(
                record_id=str(row['record_id']),
                weekly_work_hours=float(row['weekly_work_hours']),
                stress_level=str(row['stress_level'])
            )
            db.add(nueva_operativa)
            
            # PARTE C: Cultura y Retención (Fuente 3)
            nueva_cultura = models.CulturaYRetencion(
                record_id=str(row['record_id']),
                employer_support_level=str(row['employer_support_level']),
                mental_health_policy_exists=str(row['mental_health_policy_exists']),
                eap_available=str(row['eap_available']),
                used_eap=str(row['used_eap']),
                workplace_stigma_felt=str(row['workplace_stigma_felt']),
                manager_support_score=float(row['manager_support_score']),
                team_collaboration_score=float(row['team_collaboration_score']),
                intention_to_leave=str(row['intention_to_leave']),
                productivity_score=float(row['productivity_score']),
                absenteeism_days_per_year=int(row['absenteeism_days_per_year'])
            )
            db.add(nueva_cultura)
        
        # Confirmación de la transacción masiva
        db.commit()
        print("[ETL] Pipeline finalizado con éxito de forma reproducible. Datos cargados en Docker.")
        
    except Exception as e:
        db.rollback()
        print(f"[ERROR DURANTE LA CARGA] Se hizo un rollback preventivo. Detalles: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    ejecutar_pipeline()