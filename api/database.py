# api/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Configuración de credenciales apuntando al contenedor Docker
# NOTA: Usamos 'localhost' porque la API correrá de forma local apuntando al puerto expuesto de Docker
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123_duoc")  # La contraseña que definimos en Docker
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "db_salud_laboral")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Crear el motor de conexión de SQLAlchemy
engine = create_engine(DATABASE_URL)

# Configurar la fábrica de sesiones para interactuar con las tablas
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para mapear los modelos de Python a las tablas SQL
Base = declarative_base()

# Dependencia para abrir y cerrar la sesión de la base de datos automáticamente
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()