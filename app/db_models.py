from app.db_connection import Base
from sqlalchemy import String, Float, SmallInteger, Date, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional

class Elemento(Base):
    __tablename__ = 'elementos'
    
    id_elemento: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elemento: Mapped[str] = mapped_column(String(10), nullable=False)
    nombre_elemento: Mapped[str] = mapped_column(String(85), nullable=False)
    unidad_medicion: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    significado_unidad: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    mediciones: Mapped[list["Medicion"]] = relationship("Medicion", back_populates="elemento")

class Estacion(Base):
    __tablename__ = 'estaciones'
    
    id_estacion: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clave_estacion: Mapped[str] = mapped_column(String(10), nullable=False)
    nombre_estacion: Mapped[str] = mapped_column(String(35), nullable=False)
    delegacion_municipio: Mapped[str] = mapped_column(String(65), nullable=False)
    entidad: Mapped[str] = mapped_column(String(45), nullable=False)
    estatus: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    domicilio: Mapped[Optional[str]] = mapped_column(String(285), nullable=True)
    latitud: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitud: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    altitud: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    unidad: Mapped[Optional[str]] = mapped_column(String(285), nullable=True)
    validado: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    mediciones: Mapped[list["Medicion"]] = relationship("Medicion", back_populates="estacion")

class Medicion(Base):
    __tablename__ = 'mediciones'
    
    id_medicion: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_estacion: Mapped[Optional[int]] = mapped_column(
        Integer, 
        ForeignKey('estaciones.id_estacion', ondelete='CASCADE', onupdate='CASCADE'), 
        nullable=False
    )
    id_elemento: Mapped[Optional[int]] = mapped_column(
        Integer, 
        ForeignKey('elementos.id_elemento', ondelete='CASCADE', onupdate='CASCADE'), 
        nullable=False
    )
    medicion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fecha: Mapped[Optional[Date]] = mapped_column(Date, nullable=True)
    hora: Mapped[int] = mapped_column(Integer, nullable=False)
    dia: Mapped[int] = mapped_column(Integer, nullable=False)
    dayWeek: Mapped[int] = mapped_column(Integer, nullable=False)
    mes: Mapped[int] = mapped_column(Integer, nullable=False)
    anio: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relaciones con tipado mejorado
    estacion: Mapped["Estacion"] = relationship("Estacion", back_populates="mediciones")
    elemento: Mapped["Elemento"] = relationship("Elemento", back_populates="mediciones")