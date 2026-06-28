# etl.py
import os
import pandas as pd
from sqlalchemy.orm import Session
import sys

# Agregamos la raíz del proyecto al path para poder importar desde la carpeta api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.database import engine, SessionLocal
import api.models as models

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mental_health_workplace (1).csv")

def conectar_db():
    """Intenta conectar con la base de datos en Docker con manejo de errores avanzado."""
    try:
        db = SessionLocal()
        # CORRECCIÓN: Usamos una consulta nativa universal que no requiere tablas existentes
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        print("[ETL] Conexión exitosa a la base de datos en Docker.")
        return db
    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudo conectar a la base de datos en PostgreSQL: {e}")
        print("[CONSEJO] Asegúrate de que el contenedor de Docker esté encendido (docker-compose up -d).")
        sys.exit(1)

def ejecutar_pipeline_etl():
    print("[ETL] Iniciando Pipeline de forma automatizada...")
    
    # --- 1. EXTRACCIÓN (Extract) ---
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] No se encontró el archivo de datos original en: {CSV_PATH}")
        sys.exit(1)
        
    print(f"[ETL] Leyendo dataset maestro desde {CSV_PATH}...")
    df_maestro = pd.read_csv(CSV_PATH)
    
    # --- 2. TRANSFORMACIÓN (Transform) ---
    print("[ETL] Aplicando transformaciones robustas y limpieza...")
    
    # Reemplazar valores nulos específicos en strings con etiquetas comprensibles para negocio
    # Esto evita problemas con tipos de datos estrictos en la BD
    df_maestro['employer_support_level'] = df_maestro['employer_support_level'].fillna('Not Specified')
    df_maestro['used_eap'] = df_maestro['used_eap'].fillna('No')
    df_maestro['workplace_stigma_felt'] = df_maestro['workplace_stigma_felt'].fillna('None')
    
    # Limpieza estándar de textos (quitar espacios en blanco invisibles si existen)
    for col in df_maestro.select_dtypes(include=['object']).columns:
        df_maestro[col] = df_maestro[col].astype(str).str.strip()

    # --- SIMULACIÓN DE DIVISIÓN DE LAS 3 FUENTES (Exigencia de Pauta) ---
    # Fuente 1 y 2 (Se guardarían en sistemas externos o archivos temporales, el dashboard las unirá después)
    print("[ETL] Simulando extracción y aislamiento de Fuente 1 (Demográficos CSV) y Fuente 2 (Clínicos API)...")
    
    # Fuente 3: Seleccionamos estrictamente la tercera parte que va a nuestra Base de Datos en PostgreSQL
    df_postgres = df_maestro[[
        'record_id', 'company_size', 'work_model', 'weekly_work_hours', 
        'weekly_overtime_hours', 'remote_work_preference', 'employer_support_level', 
        'mental_health_policy_exists', 'eap_available', 'used_eap', 
        'workplace_stigma_felt', 'manager_support_score', 'team_collaboration_score', 
        'intention_to_leave', 'productivity_score', 'absenteeism_days_per_year'
    ]].copy()

    # --- 3. CARGA (Load) ---
    db = conectar_db()
    
    try:
        print("[ETL] Limpiando tablas previas mediante recreación del esquema...")
        # Volvemos a crear las tablas limpias para asegurar la REPRODUCIBILIDAD total del script
        models.Base.metadata.drop_all(bind=engine)
        models.Base.metadata.create_all(bind=engine)
        
        print(f"[ETL] Insertando {len(df_postgres)} registros normalizados en PostgreSQL...")
        
        # Inserción masiva optimizada por bloques (bulk insert) para grandes volúmenes de datos
        objetos_empleados = []
        objetos_operativas = []
        objetos_cultura = []
        
        for _, fila in df_postgres.iterrows():
            # Registro en tabla: empleados_base
            emp = models.EmpleadoBase(
                record_id=fila['record_id'],
                company_size=fila['company_size']
            )
            objetos_empleados.append(emp)
            
            # Registro en tabla: condiciones_operativas
            op = models.CondicionOperativa(
                record_id=fila['record_id'],
                work_model=fila['work_model'],
                weekly_work_hours=int(fila['weekly_work_hours']),
                weekly_overtime_hours=int(fila['weekly_overtime_hours']),
                remote_work_preference=fila['remote_work_preference']
            )
            objetos_operativas.append(op)
            
            # Registro en tabla: cultura_y_retencion
            cul = models.CulturaYRetencion(
                record_id=fila['record_id'],
                employer_support_level=fila['employer_support_level'],
                mental_health_policy_exists=fila['mental_health_policy_exists'],
                eap_available=fila['eap_available'],
                used_eap=fila['used_eap'],
                workplace_stigma_felt=fila['workplace_stigma_felt'],
                manager_support_score=float(fila['manager_support_score']),
                team_collaboration_score=float(fila['team_collaboration_score']),
                intention_to_leave=fila['intention_to_leave'],
                productivity_score=float(fila['productivity_score']),
                absenteeism_days_per_year=int(fila['absenteeism_days_per_year'])
            )
            objetos_cultura.append(cul)
            
        # Ejecutar las cargas en la sesión de la base de datos
        db.bulk_save_objects(objetos_empleados)
        db.bulk_save_objects(objetos_operativas)
        db.bulk_save_objects(objetos_cultura)
        
        # Confirmar y guardar la transacción en PostgreSQL
        db.commit()
        print("[ETL] Pipeline finalizado con éxito de forma reproducible. Datos cargados en Docker.")
        
    except Exception as e:
        db.rollback()
        print(f"[ERROR DURANTE LA CARGA] Se hizo un rollback preventivo. Detalles: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    ejecutar_pipeline_etl()