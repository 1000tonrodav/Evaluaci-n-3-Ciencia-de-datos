# api/models.py
from sqlalchemy import Column, String, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EmpleadoBase(Base):
    __tablename__ = "empleados_base"

    record_id = Column(String(50), primary_key=True, index=True)
    company_size = Column(String(50), nullable=False)
    country = Column(String(100), nullable=False)

    operativas = relationship("CondicionOperativa", back_populates="empleado", cascade="all, delete-orphan")
    cultura = relationship("CulturaYRetencion", back_populates="empleado", cascade="all, delete-orphan")


class CondicionOperativa(Base):
    __tablename__ = "condiciones_operativas"

    record_id = Column(String(50), ForeignKey("empleados_base.record_id"), primary_key=True)
    weekly_work_hours = Column(Float, nullable=False)
    stress_level = Column(String(50), nullable=False) # <-- Aquí nos aseguramos de que acepte 'stress_level'

    empleado = relationship("EmpleadoBase", back_populates="operativas")


class CulturaYRetencion(Base):
    __tablename__ = "cultura_y_retencion"

    record_id = Column(String(50), ForeignKey("empleados_base.record_id"), primary_key=True)
    employer_support_level = Column(String(50), nullable=False)
    mental_health_policy_exists = Column(String(50), nullable=False)
    eap_available = Column(String(50), nullable=False)
    used_eap = Column(String(50), nullable=False)
    workplace_stigma_felt = Column(String(50), nullable=False)
    manager_support_score = Column(Float, nullable=False)
    team_collaboration_score = Column(Float, nullable=False)
    intention_to_leave = Column(String(50), nullable=False)
    productivity_score = Column(Float, nullable=False)
    absenteeism_days_per_year = Column(Integer, nullable=False)

    empleado = relationship("EmpleadoBase", back_populates="cultura")