from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import date

class ElementoResponse(BaseModel):
    """
    Schema para la respuesta de elementos.
    
    Representa la estructura de datos que se retorna
    cuando se consultan elementos del sistema.
    """
    id_elemento: int
    elemento: str
    nombre_elemento: str
    unidad_medicion: Optional[str]
    significado_unidad: Optional[str]

    model_config = ConfigDict(from_attributes=True)

class EstacionResponse(BaseModel):
    """
    Schema para la respuesta de estaciones.
    
    Representa la estructura de datos que se retorna
    cuando se consultan estaciones del sistema.
    """
    id_estacion: int
    clave_estacion: str
    nombre_estacion: str
    delegacion_municipio: str
    entidad: str
    estatus: Optional[str]
    domicilio: Optional[str]
    latitud: Optional[float]
    longitud: Optional[float]
    altitud: Optional[int]
    unidad: Optional[str]
    validado: Optional[int]

    model_config = ConfigDict(from_attributes=True)

class FechaRangeResponse(BaseModel):
    """
    Schema para la respuesta del rango de fechas.
    
    Representa la fecha mínima y máxima encontrada
    en las mediciones del sistema.
    """
    fecha_minima: Optional[date]
    fecha_maxima: Optional[date]
    total_registros: int

    model_config = ConfigDict(from_attributes=True)

class MedicionTimeSeriesResponse(BaseModel):
    datetime: str
    medicion: float