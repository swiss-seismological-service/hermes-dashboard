from functools import lru_cache
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    WEBSERVICE_URL: str
    PROJECT_ID: UUID


@lru_cache()
def get_config():
    return Config()
