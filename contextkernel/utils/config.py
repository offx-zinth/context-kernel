from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")

class Neo4jConfig(BaseSettings):
    uri: str = Field(default="neo4j://localhost:7687", env="NEO4J_URI")
    user: str = Field(default="neo4j", env="NEO4J_USER")
    password: str = Field(default="password", env="NEO4J_PASSWORD")

class VectorDBConfig(BaseSettings):
    type: str = Field(default="pinecone", env="VECTOR_DB_TYPE")
    api_key: Optional[str] = Field(default=None, env="VECTOR_DB_API_KEY")
    environment: Optional[str] = Field(default=None, env="VECTOR_DB_ENVIRONMENT")

class EmbeddingConfig(BaseSettings):
    model_name: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL_NAME")
    api_key: Optional[str] = Field(default=None, env="EMBEDDING_API_KEY") # For services like OpenAI

class NLPServiceConfig(BaseSettings):
    provider: str = Field(default="openai", env="NLP_SERVICE_PROVIDER")
    api_key: Optional[str] = Field(default=None, env="NLP_SERVICE_API_KEY")
    model: Optional[str] = Field(default="gpt-3.5-turbo", env="NLP_SERVICE_MODEL")

class S3Config(BaseSettings):
    bucket_name: str = Field(default="my-context-kernel-bucket", env="S3_BUCKET_NAME")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    region_name: Optional[str] = Field(default="us-east-1", env="AWS_REGION_NAME")

class FileSystemConfig(BaseSettings):
    base_path: str = Field(default="/tmp/context_kernel_data", env="FILESYSTEM_BASE_PATH")

class AppSettings(BaseSettings):
    redis_config: RedisConfig = RedisConfig()
    neo4j_config: Neo4jConfig = Neo4jConfig()
    vector_db_config: VectorDBConfig = VectorDBConfig()
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    nlp_service_config: NLPServiceConfig = NLPServiceConfig()
    s3_config: S3Config = S3Config()
    filesystem_config: FileSystemConfig = FileSystemConfig()

    # Example of how to load a specific config if needed, e.g., for LTM raw content store
    # raw_content_store_type: str = Field(default="filesystem", env="RAW_CONTENT_STORE_TYPE") # 's3' or 'filesystem'

    class Config:
        env_nested_delimiter = '__' # e.g. REDIS_CONFIG__HOST

if __name__ == "__main__":
    # Example usage:
    # Set environment variables before running this script, e.g.:
    # export REDIS_HOST=myredishost
    # export NEO4J_PASSWORD=securepassword
    # export VECTOR_DB_API_KEY=abcdef12345
    # ... and so on for other settings

    settings = AppSettings()

    print("Redis Config:", settings.redis_config)
    print("Neo4j Config:", settings.neo4j_config)
    print("Vector DB Config:", settings.vector_db_config)
    print("Embedding Config:", settings.embedding_config)
    print("NLP Service Config:", settings.nlp_service_config)
    print("S3 Config:", settings.s3_config)
    print("File System Config:", settings.filesystem_config)

    # To access a specific setting:
    print(f"\nRedis Host: {settings.redis_config.host}")
    print(f"Neo4j User: {settings.neo4j_config.user}")

    # Example of how environment variables override defaults:
    # Assumes REDIS_HOST is set in the environment
    # import os
    # os.environ["REDIS_HOST"] = "env_redis_host"
    # os.environ["APP_REDIS_CONFIG__PORT"] = "1234" # For nested model
    # settings_env_override = AppSettings()
    # print(f"\nRedis Host (from env): {settings_env_override.redis_config.host}")
    # print(f"Redis Port (from env with nested): {settings_env_override.redis_config.port}")
