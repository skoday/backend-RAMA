from fastapi import APIRouter, status, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from app.db_connection import get_db
from sqlalchemy import func, select, text
from datetime import datetime
from app import db_models
from app.schemas import (
    ElementoResponse, EstacionResponse, FechaRangeResponse,
    MedicionTimeSeriesResponse, PredictionResult,
    HistoricalDataPoint, PredictionResponse)
from app.lstm import get_predictor, train_model_for_station_element
import pandas as pd

app = APIRouter(
    prefix='',
    tags=['resources']
)

# Endpoint para recuperar todos los elementos
@app.get(
    '/elementos',
    response_model=list[ElementoResponse],
    status_code=status.HTTP_200_OK,
    summary="Obtener todos los elementos",
    description="Recupera una lista completa de todos los elementos disponibles en el sistema"
)
def get_all_elementos(db: Session = Depends(get_db)):
    """
    Recupera todos los elementos de la base de datos.
    
    Returns:
        List[ElementoResponse]: Lista de todos los elementos
    
    Raises:
        HTTPException: 404 si no se encuentran elementos
        HTTPException: 500 si hay error en la base de datos
    """
    try:
        elementos = db.query(db_models.Elemento).all()
        
        if not elementos:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontraron elementos en el sistema"
            )
        
        return elementos
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

# Endpoint para recuperar todas las estaciones
@app.get(
    '/estaciones',
    response_model=list[EstacionResponse],
    status_code=status.HTTP_200_OK,
    summary="Obtener todas las estaciones",
    description="Recupera una lista completa de todas las estaciones disponibles en el sistema"
)
def get_all_estaciones(db: Session = Depends(get_db)):
    """
    Recupera todas las estaciones de la base de datos.
    
    Returns:
        List[EstacionResponse]: Lista de todas las estaciones
    
    Raises:
        HTTPException: 404 si no se encuentran estaciones
        HTTPException: 500 si hay error en la base de datos
    """
    try:
        estaciones = db.query(db_models.Estacion).all()
        
        if not estaciones:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontraron estaciones en el sistema"
            )
        
        return estaciones
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get(
    '/mediciones/rango-fechas',
    response_model=FechaRangeResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtener rango de fechas de mediciones",
    description="Recupera la fecha mínima y máxima de todas las mediciones en el sistema"
)
def get_fecha_range_mediciones(db: Session = Depends(get_db)):
    """
    Obtiene el rango de fechas (mínima y máxima) de las mediciones.
    
    Returns:
        FechaRangeResponse: Objeto con fecha_minima, fecha_maxima y total_registros
    
    Raises:
        HTTPException: 404 si no se encuentran mediciones
        HTTPException: 500 si hay error en la base de datos
    """
    try:
        # Query usando SQLAlchemy v2 con func.min() y func.max()
        result = db.execute(
            select(
                func.min(db_models.Medicion.fecha).label('fecha_minima'),
                func.max(db_models.Medicion.fecha).label('fecha_maxima'),
                func.count(db_models.Medicion.id_medicion).label('total_registros')
            )
        ).first()
        
        if not result or result.total_registros == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontraron mediciones en el sistema"
            )
        
        return FechaRangeResponse(
            fecha_minima=result.fecha_minima,
            fecha_maxima=result.fecha_maxima,
            total_registros=result.total_registros
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get(
    '/mediciones/time-series',
    response_model=list[MedicionTimeSeriesResponse],
    status_code=status.HTTP_200_OK,
    summary="Obtener mediciones por estación y elemento",
    description="Recupera las mediciones de una estación específica y elemento ordenadas por fecha y hora"
)
def get_mediciones_time_series(
    id_estacion: int = Query(..., description="ID de la estación"),
    id_elemento: int = Query(..., description="ID del elemento"),
    db: Session = Depends(get_db)
):
    """
    Obtiene las mediciones de una estación y elemento específicos ordenadas cronológicamente.
    
    Args:
        id_estacion: ID de la estación
        id_elemento: ID del elemento
        db: Sesión de base de datos
    
    Returns:
        List[MedicionTimeSeriesResponse]: Lista de mediciones con datetime y valor
    
    Raises:
        HTTPException: 404 si no se encuentran mediciones
        HTTPException: 500 si hay error en la base de datos
    """
    try:
        # La query SQL que creamos antes, adaptada para SQLAlchemy
        query = text("""
            SELECT 
                fecha + INTERVAL '1 hour' * hora AS datetime,
                medicion
            FROM mediciones
            WHERE id_estacion = :id_estacion 
                AND id_elemento = :id_elemento
                AND medicion >= 0
            ORDER BY fecha, hora;
        """)
        
        result = db.execute(query, {
            'id_estacion': id_estacion,
            'id_elemento': id_elemento
        }).fetchall()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontraron mediciones para estación {id_estacion} y elemento {id_elemento}"
            )
        
        # Convertir a formato de respuesta
        mediciones = []
        for row in result:
            mediciones.append({
                'datetime': row.datetime.isoformat(),
                'medicion': float(row.medicion)
            })
        
        return mediciones
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )
    

@app.get(
    '/mediciones/predict',
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Generar predicciones de series de tiempo",
    description="Genera predicciones para una estación y elemento específicos usando LSTM"
)
def predict_time_series(
    id_estacion: int = Query(..., description="ID de la estación"),
    id_elemento: int = Query(..., description="ID del elemento"),
    start_date: str = Query(..., description="Fecha de inicio (YYYY-MM-DD HH:MM:SS)"),
    end_date: str = Query(..., description="Fecha de fin (YYYY-MM-DD HH:MM:SS)"),
    prediction_window: int = Query(..., description="Ventana de predicción en horas", ge=1, le=720),
    db: Session = Depends(get_db)
):
    """
    Genera predicciones de series de tiempo usando LSTM.
    
    Args:
        id_estacion: ID de la estación
        id_elemento: ID del elemento  
        start_date: Fecha de inicio en formato YYYY-MM-DD HH:MM:SS
        end_date: Fecha de fin en formato YYYY-MM-DD HH:MM:SS
        prediction_window: Número de horas a predecir (máximo 720)
        db: Sesión de base de datos
        
    Returns:
        PredictionResult: Resultado con predicciones y metadatos
    """
    try:
        # Validate dates
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato de fecha inválido. Use YYYY-MM-DD HH:MM:SS"
            )
        
        if start_dt >= end_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La fecha de inicio debe ser anterior a la fecha de fin"
            )
        
        # Get data from database with date filtering
        query = text("""
            SELECT 
                fecha + INTERVAL '1 hour' * hora AS datetime,
                medicion
            FROM mediciones
            WHERE id_estacion = :id_estacion 
                AND id_elemento = :id_elemento
                AND medicion >= 0
                AND (fecha + INTERVAL '1 hour' * hora) BETWEEN :start_date AND :end_date
            ORDER BY fecha, hora;
        """)
        
        result = db.execute(query, {
            'id_estacion': id_estacion,
            'id_elemento': id_elemento,
            'start_date': start_dt,
            'end_date': end_dt
        }).fetchall()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontraron mediciones para los parámetros especificados"
            )
        
        # Convert to DataFrame
        data = []
        for row in result:
            data.append({
                'datetime': row.datetime,
                'medicion': float(row.medicion)
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Get predictor and make predictions
        predictor = train_model_for_station_element(df)
        predictions, timestamps = predictor.predict(df, prediction_window)
        
        # Format historical data response
        historical_responses = []
        for idx, row in df.iterrows():
            historical_responses.append(HistoricalDataPoint(
                timestamp=idx.isoformat(),
                actual_value=float(row['medicion'])
            ))
        
        # Format prediction response
        prediction_responses = []
        for pred, ts in zip(predictions, timestamps):
            prediction_responses.append(PredictionResponse(
                timestamp=ts,
                predicted_value=float(pred)
            ))
        
        return PredictionResult(
            historical_data=historical_responses,  # NEW: Include historical data
            predictions=prediction_responses,
            model_info=predictor.get_model_info(),
            data_info={
                "input_records": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "station_id": id_estacion,
                "element_id": id_elemento,
                "prediction_window_hours": prediction_window,
                "historical_data_points": len(historical_responses)  # NEW: Count of historical points
            },
            success=True,
            message=f"Predicción generada exitosamente para {prediction_window} horas con {len(historical_responses)} puntos históricos"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error de validación: {str(e)}"
        )
    except Exception as e:
        #logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando predicción: {str(e)}"
        )