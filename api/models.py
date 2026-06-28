# api/models.py
from sqlalchemy import Column, String, Integer, Numeric, ForeignKey
from sqlalchemy.orm import relationship
from api.database import Base

class EmpleadoBase(Base):
    __tablename__ = "empleados_base"

    record_id = Column(String(50), primary_key=True, index=True)
    company_size = Column(String(50), nullable=False)

    # Relaciones que permiten hacer consultas relacionales (JOINs) fácilmente desde Python
    operativas = relationship("CondicionOperativa", back_populates="empleado", cascade="all, delete-orphan")
    cultura = relationship("CulturaYRetencion", back_populates="empleado", cascade="all, delete-orphan")


class CondicionOperativa(Base):
    __tablename__ = "condiciones_operativas"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    record_id = Column(String(50), ForeignKey("empleados_base.record_id", ondelete="CASCADE"), nullable=False)
    work_model = Column(String(50), nullable=False)
    weekly_work_hours = Column(Integer, nullable=False)
    weekly_overtime_hours = Column(Integer, nullable=False)
    remote_work_preference = Column(String(100), nullable=False)

    empleado = relationship("EmpleadoBase", back_populates="operativas")


class CulturaYRetencion(Base):
    __tablename__ = "cultura_y_retencion"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    record_id = Column(String(50), ForeignKey("empleados_base.record_id", ondelete="CASCADE"), nullable=False)
    employer_support_level = Column(String(50), nullable=True)  # Corregido: Comentario en Python
    mental_health_policy_exists = Column(String(50), nullable=False)
    eap_available = Column(String(50), nullable=False)
    used_eap = Column(String(50), nullable=True)                # Corregido: Comentario en Python
    workplace_stigma_felt = Column(String(50), nullable=True)          # Corregido: Comentario en Python
    manager_support_score = Column(Numeric(4, 2), nullable=False)
    team_collaboration_score = Column(Numeric(4, 2), nullable=False)
    intention_to_leave = Column(String(50), nullable=False)
    productivity_score = Column(Numeric(4, 2), nullable=False)
    absenteeism_days_per_year = Column(Integer, nullable=False)

    empleado = relationship("EmpleadoBase", back_populates="cultura")