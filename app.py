"""
FastAPI endpoints for California Housing Model

Recall the model takes in housing district info
and returns the average housing value in 100_000 USD

"""
from joblib import load
from fastapi import FastAPI, Query
import pandas as pd
from pydantic import BaseModel, Field
from src.main import restore_model

OBS_BOUNDS = load("models/observation_bounds.joblib")

MODEL = restore_model()

app = FastAPI()


class DistrictInfo(BaseModel):
    """class for housing district info"""
    med_inc: float = Field(
        ge=OBS_BOUNDS['min']['MedInc'],
        le=OBS_BOUNDS['max']['MedInc'],
        default=OBS_BOUNDS['mean']['MedInc'])
    house_age: float = Field(
        ge=OBS_BOUNDS['min']['HouseAge'],
        le=OBS_BOUNDS['max']['HouseAge'],
        default=OBS_BOUNDS['mean']['HouseAge'])
    ave_rooms: float = Field(
        ge=OBS_BOUNDS['min']['AveRooms'],
        le=OBS_BOUNDS['max']['AveRooms'],
        default=OBS_BOUNDS['mean']['AveRooms'])
    ave_bedrms: float = Field(
        ge=OBS_BOUNDS['min']['AveBedrms'],
        le=OBS_BOUNDS['max']['AveBedrms'],
        default=OBS_BOUNDS['mean']['AveBedrms'])
    population: float = Field(
        ge=OBS_BOUNDS['min']['Population'],
        le=OBS_BOUNDS['max']['Population'],
        default=OBS_BOUNDS['mean']['Population'])
    ave_occup: float = Field(
        ge=OBS_BOUNDS['min']['AveOccup'],
        le=OBS_BOUNDS['max']['AveOccup'],
        default=OBS_BOUNDS['mean']['AveOccup'])
    latitude: float = Field(
        ge=OBS_BOUNDS['min']['Latitude'],
        le=OBS_BOUNDS['max']['Latitude'],
        default=OBS_BOUNDS['mean']['Latitude'])
    longitude: float = Field(
        ge=OBS_BOUNDS['min']['Longitude'],
        le=OBS_BOUNDS['max']['Longitude'],
        default=OBS_BOUNDS['mean']['Longitude'])


def format_and_convert_user_input(
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude):
    """take inputs and return df"""
    return pd.DataFrame(
        {"MedInc": [med_inc],
         "HouseAge": [house_age],
         "AveRooms": [ave_rooms],
         "AveBedrms": [ave_bedrms],
         "Population": [population],
         "AveOccup": [ave_occup],
         "Latitude": [latitude],
         "Longitude": [longitude]})


def return_prediction(df, model=MODEL):
    """return prediction in real dollars"""
    return {"prediction":
            round(model.predict(df)[0] * 100_000, 2)}


@app.get("/")
async def root():
    """return hello world as a greeting"""

    return {"greeting": "Hello World"}


@app.post("/housing_post")
async def housing_post(dist_info: DistrictInfo):
    """return prediction from json packet"""
    obs = format_and_convert_user_input(
        med_inc=dist_info.med_inc,
        house_age=dist_info.house_age,
        ave_rooms=dist_info.ave_rooms,
        ave_bedrms=dist_info.ave_bedrms,
        population=dist_info.population,
        ave_occup=dist_info.ave_occup,
        latitude=dist_info.latitude,
        longitude=dist_info.longitude)

    # you could try
    # dist_info.model_dump()

    return return_prediction(obs)


@app.get("/query")
async def housing_query(
        med_inc: float = Query(
            ge=OBS_BOUNDS['min']['MedInc'],
            le=OBS_BOUNDS['max']['MedInc'],
            default=OBS_BOUNDS['mean']['MedInc']),
        house_age: float = Query(
            ge=OBS_BOUNDS['min']['HouseAge'],
            le=OBS_BOUNDS['max']['HouseAge'],
            default=OBS_BOUNDS['mean']['HouseAge']),
        ave_rooms: float = Query(
            ge=OBS_BOUNDS['min']['AveRooms'],
            le=OBS_BOUNDS['max']['AveRooms'],
            default=OBS_BOUNDS['mean']['AveRooms']),
        ave_bedrms: float = Query(
            ge=OBS_BOUNDS['min']['AveBedrms'],
            le=OBS_BOUNDS['max']['AveBedrms'],
            default=OBS_BOUNDS['mean']['AveBedrms']),
        population: float = Query(
            ge=OBS_BOUNDS['min']['Population'],
            le=OBS_BOUNDS['max']['Population'],
            default=OBS_BOUNDS['mean']['Population']),
        ave_occup: float = Query(
            ge=OBS_BOUNDS['min']['AveOccup'],
            le=OBS_BOUNDS['max']['AveOccup'],
            default=OBS_BOUNDS['mean']['AveOccup']),
        latitude: float = Query(
            ge=OBS_BOUNDS['min']['Latitude'],
            le=OBS_BOUNDS['max']['Latitude'],
            default=OBS_BOUNDS['mean']['Latitude']),
        longitude: float = Query(
            ge=OBS_BOUNDS['min']['Longitude'],
            le=OBS_BOUNDS['max']['Longitude'],
            default=OBS_BOUNDS['mean']['Longitude'])):
    """take query parameters and return housing prediction"""

    obs = format_and_convert_user_input(
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude)

    return return_prediction(obs)


if __name__ == "__main__":

    print(OBS_BOUNDS)
    print(OBS_BOUNDS['min']['MedInc'])
