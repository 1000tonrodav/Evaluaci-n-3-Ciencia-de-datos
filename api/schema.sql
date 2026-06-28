-- api/schema.sql

-- 1. Limpieza de tablas previas (Garantiza la REPRODUCIBILIDAD si reinicias el contenedor)
DROP TABLE IF EXISTS cultura_y_retencion CASCADE;
DROP TABLE IF EXISTS condiciones_operativas CASCADE;
DROP TABLE IF EXISTS empleados_base CASCADE;

-- 2. Tabla Principal en la BD (Punto de anclaje para las llaves foráneas)
CREATE TABLE empleados_base (
    record_id VARCHAR(50) PRIMARY KEY,
    company_size VARCHAR(50) NOT NULL
);

-- 3. Tabla Operativa (Variables de carga de trabajo de la BD)
CREATE TABLE condiciones_operativas (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(50) NOT NULL,
    work_model VARCHAR(50) NOT NULL,
    weekly_work_hours INT NOT NULL,
    weekly_overtime_hours INT NOT NULL,
    remote_work_preference VARCHAR(100) NOT NULL,
    CONSTRAINT fk_empleado_operativas 
        FOREIGN KEY (record_id) 
        REFERENCES empleados_base(record_id) 
        ON DELETE CASCADE
);

-- 4. Tabla de Clima, Desempeño y Retención (Variables de negocio)
CREATE TABLE cultura_y_retencion (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(50) NOT NULL,
    employer_support_level VARCHAR(50),          -- Puede ser NULL según el archivo original
    mental_health_policy_exists VARCHAR(50) NOT NULL,
    eap_available VARCHAR(50) NOT NULL,
    used_eap VARCHAR(50),                         -- Puede ser NULL según el archivo original
    workplace_stigma_felt VARCHAR(50),            -- Puede ser NULL según el archivo original
    manager_support_score NUMERIC(4, 2) NOT NULL,
    team_collaboration_score NUMERIC(4, 2) NOT NULL,
    intention_to_leave VARCHAR(50) NOT NULL,      -- Nuestra futura variable objetivo
    productivity_score NUMERIC(4, 2) NOT NULL,
    absenteeism_days_per_year INT NOT NULL,
    CONSTRAINT fk_empleado_cultura 
        FOREIGN KEY (record_id) 
        REFERENCES empleados_base(record_id) 
        ON DELETE CASCADE
);

-- 5. Índices de optimización (Pauta: "Optimización para grandes volúmenes")
CREATE INDEX idx_operativas_record ON condiciones_operativas(record_id);
CREATE INDEX idx_cultura_record ON cultura_y_retencion(record_id);
CREATE INDEX idx_cultura_leave ON cultura_y_retencion(intention_to_leave);