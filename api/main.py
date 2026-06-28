# api/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import api.models as models
from api.database import get_db, engine

# Inicializar FastAPI con documentación automática habilitada
app = FastAPI(
    title="API de Salud Laboral y Retención - Duoc UC",
    description="Endpoints para disponibilizar la Fuente 3 de datos almacenada en PostgreSQL Docker",
    version="1.0.0"
)

# Crear las tablas en la BD si por alguna razón no existieran al levantar la API
models.Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {
        "status": "activa",
        "proyecto": "Sistema de Monitoreo de Salud Laboral",
        "documentacion": "/docs"
    }

@app.get("/api/cultura-retencion", response_model=list)
def obtener_datos_cultura(limit: int = 100, db: Session = Depends(get_db)):
    """
    Endpoint que retorna los datos de Clima, Desempeño y Retención (Fuente 3).
    Permite paginación mediante el parámetro 'limit' para optimizar el rendimiento.
    """
    try:
        # Consultamos directamente la tabla cultura_y_retencion mapeada en el modelo
        registros = db.query(models.CulturaYRetencion).limit(limit).all()
        
        # Formateamos la respuesta en un JSON limpio para que el Dashboard lo consuma sin problemas
        resultado = []
        for r in registros:
            resultado.append({
                "record_id": r.record_id,
                "employer_support_level": r.employer_support_level,
                "mental_health_policy_exists": r.mental_health_policy_exists,
                "eap_available": r.eap_available,
                "used_eap": r.used_eap,
                "workplace_stigma_felt": r.workplace_stigma_felt,
                "manager_support_score": float(r.manager_support_score),
                "team_collaboration_score": float(r.team_collaboration_score),
                "intention_to_leave": r.intention_to_leave,  # Nuestra variable objetivo
                "productivity_score": float(r.productivity_score),
                "absenteeism_days_per_year": r.absenteeism_days_per_year
            })
        return resultado
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al consultar la base de datos: {str(e)}")