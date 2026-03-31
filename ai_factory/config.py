import os
from urllib.parse import urlparse, urlencode, parse_qs
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    simulation_mode: bool = True
    database_url: str = os.environ.get("DATABASE_URL", "")
    async_database_url: str = ""
    host: str = "0.0.0.0"
    port: int = 8000

    def model_post_init(self, __context):
        if self.database_url and not self.async_database_url:
            url = self.database_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            params.pop("sslmode", None)
            clean_query = urlencode(params, doseq=True)
            url = parsed._replace(query=clean_query).geturl()
            self.async_database_url = url

    model_config = {"env_prefix": "", "extra": "ignore"}


settings = Settings()
