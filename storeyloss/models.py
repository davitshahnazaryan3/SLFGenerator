import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Union, List


class ComponentDataModel(BaseModel):
    ITEM: Optional[int] = None
    ID: Optional[str] = None
    EDP: str
    Component: str
    Group: Optional[int] = None
    Quantity: float
    Damage_States: int = Field(alias="Damage States")

    @validator('ITEM')
    def validate_id(cls, v):
        if v is not None and v < 0:
            raise ValueError('ITEM ID must not be below zero')
        return v

    @validator('Group', 'ITEM', pre=True)
    def allow_none(cls, v):
        if v is None or np.isnan(v):
            return None
        else:
            return v


class CorrelationTreeModel(BaseModel):
    ITEM: int
    dependent_on_item: str = Field(alias="DEPENDANT ON ITEM")

    @validator('ITEM')
    def validate_id(cls, v):
        if v < 0:
            raise ValueError('ITEM ID must not be below zero')
        return v


class ItemBase(BaseModel):
    RootModel: Dict[str, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class ItemsModel(BaseModel):
    RootModel: Dict[int, ItemBase]


class FragilityModel(BaseModel):
    EDP: np.ndarray
    ITEMs: ItemsModel

    class Config:
        arbitrary_types_allowed = True


class DamageStateModel(BaseModel):
    RootModel: Dict[int, Dict[int, np.ndarray]]

    class Config:
        arbitrary_types_allowed = True


class DamageStateValidateModel(BaseModel):
    ds: DamageStateModel


class CostModel(BaseModel):
    RootModel: Dict[int, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class SimulationModel(BaseModel):
    RootModel: Dict[int, CostModel]

    class Config:
        arbitrary_types_allowed = True


class CostValidateModel(BaseModel):
    cost: CostModel


class SimulationValidateModel(BaseModel):
    sim: SimulationModel


class FittingModelBase(BaseModel):
    popt: np.ndarray
    pcov: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class FittingParametersModel(BaseModel):
    RootModel: Dict[str, FittingModelBase]


class FittedLossModel(BaseModel):
    RootModel: Dict[str, np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class LossModel(BaseModel):
    loss: Dict[int, Dict[Union[int, str], float]]
    loss_ratio: Dict[int, Dict[Union[int, str], float]]


class EALCacheModel(BaseModel):
    eal_bins: List[float] = Field(alias="eal-bins")
    iml: List[float]
    mafe: List[float]
    loss_ratio: List[float] = Field(alias="loss-ratio")


class DisEALBaseModel(BaseModel):
    eal: float
    cache: EALCacheModel


class EALBaseModel(BaseModel):
    e_nc_nd_ns: DisEALBaseModel = Field(alias="E_NC_ND_ns")
    e_nc_nd_s: DisEALBaseModel = Field(alias="E_NC_ND_s")
    e_c: DisEALBaseModel = Field(alias="E_C")
    e_nc_d: DisEALBaseModel = Field(alias="E_NC_D")
    e_lt: DisEALBaseModel = Field(alias="E_LT")

    class Config:
        arbitrary_types_allowed = True


class EALModel(BaseModel):
    RootModel: Dict[str, EALBaseModel]

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        validate_assignment = True


class DemolitionModel(BaseModel):
    median: float
    beta: float


class SLFModel(BaseModel):
    directionality: Optional[int] = Field(alias="Directionality")
    component_type: str = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(aliast="Storey")
    edp: str
    edp_range: List[float]
    slf: List[float]


class SLFPGModel(BaseModel):
    RootModel: Dict[str, SLFModel]

    class Config:
        arbitrary_types_allowed = True
