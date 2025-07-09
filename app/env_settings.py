from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env"
    )

    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str


settings = Settings(_env_file='.env')