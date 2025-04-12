# Databricks notebook source
# MAGIC %md
# MAGIC # 📄 Create an unstructured data pipeline for gen AI retrievers
# MAGIC
# MAGIC This notebook shows you how to create a data pipeline that transforms unstructured documents into a vector index. By the end of this notebook, you will have a Databricks Vector Search index that an AI agent could use to power a retriever that queries information about unstructured data.
# MAGIC
# MAGIC This notebook creates a data pipeline using the following steps:
# MAGIC 1. Download sample PDF files from the GitHub repository [Databricks demo dataset](https://github.com/databricks-demos/dbdemos-dataset/tree/main).
# MAGIC 1. Load documents into a Delta table.
# MAGIC 1. Parse documents into text strings.
# MAGIC 1. Chunk the text strings into smaller, more manageable pieces for retrieval.
# MAGIC 1. Use an embedding model to embed the chunks into vectors and store the results in a vector index.
# MAGIC
# MAGIC To learn more about building and optimizing unstructured data pipelines, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)).
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC This notebook requires Databricks Runtime Machine Learning version 14.3 and above.

# COMMAND ----------

# MAGIC %md
# MAGIC # 👉 How to use this notebook
# MAGIC
# MAGIC Follow these steps to build and refine your data pipeline's quality:
# MAGIC
# MAGIC 1. **Run this notebook to build a Vector Search index with default settings**
# MAGIC     - Configure the data source and destination tables in the `1️⃣ 📂 Data source and destination configuration` cells
# MAGIC     - Press `Run All` to create the vector index.
# MAGIC
# MAGIC     *Note: While you can adjust the other settings and modify the parsing/chunking code, we suggest doing so only after evaluating your Agent's quality so you can make improvements that specifically address root causes of quality issues.*
# MAGIC
# MAGIC 2. **Run other sample notebooks to create an AI agent retriever that queries the vector index, then evaluate the agent/retriever's quality.**
# MAGIC    - See agent examples that include boilerplate code to integrate a vector search index ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent#chat-agent-examples) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent#chat-agent-examples)).
# MAGIC
# MAGIC 3. **If the evaluation results show retrieval issues as a root cause, use this notebook to iterate on your data pipeline code & configuration.** 
# MAGIC
# MAGIC     - The following are potential fixes you can try, see Databricks documentation for debugging retrieval issues for more information ([AWS](https://docs.databricks.com/aws/generative-ai/tutorials/ai-cookbook/implementation/step-5-debug-retrieval-quality) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/tutorials/ai-cookbook/implementation/step-5-debug-retrieval-quality)).
# MAGIC       - Add missing, but relevant source documents into in the index.
# MAGIC       - Resolve any conflicting information in source documents.
# MAGIC       - Adjust the data pipeline configuration:
# MAGIC         - Modify chunk size or overlap.
# MAGIC         - Experiment with different embedding models.
# MAGIC       - Adjust the data pipeline code:
# MAGIC         - Create a custom parser or use different parsing libraries.
# MAGIC         - Develop a custom chunker or use different chunking techniques.
# MAGIC         - Extract additional metadata for each document.
# MAGIC       - Adjust the Agent's code/config in subsequent notebooks:
# MAGIC         - Change the number of documents retrieved (K).
# MAGIC         - Try a re-ranker.
# MAGIC         - Use hybrid search.
# MAGIC         - Apply extracted metadata as filters.
# MAGIC
# MAGIC **Note:** This notebook provides a foundation for creating unstructured data pipelines. For production workloads, Databricks recommends refactoring this notebook into separate components that can be orchestrated using [Databricks Workflows](https://www.databricks.com/product/workflows). In production workloads, you would pull out the code definitions into modules and separate the steps into individual tasks to be orchestrated over one or more workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Important note:** Throughout this notebook, we indicate which cells you:
# MAGIC - ✅ ✏️ *should* customize - these cells contain code and config with business logic that you should edit to meet your requirements and tune quality
# MAGIC - 🚫 ✏️ *typically should not* customize - these cells contain boilerplate code required to execute the pipeline
# MAGIC
# MAGIC Cells that don't require customization still need to be run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫 ✏️ Install Python libraries
# MAGIC
# MAGIC Only modify the following cells if you need additional packages in your code changes to the document parsing or chunking logic.
# MAGIC
# MAGIC Versions of Databricks code are not locked because Databricks ensures that changes are backward compatible.
# MAGIC Versions of open source packages are locked because package authors often make breaking changes.

# COMMAND ----------

# DBTITLE 1,Install libraries and restart Python
# MAGIC %pip install -U \
# MAGIC   "pydantic>=2.9.2" \
# MAGIC   "mlflow>=2.18.0" \
# MAGIC   "databricks-sdk" \
# MAGIC   "databricks-vectorsearch" \
# MAGIC   "pymupdf4llm==0.0.5" \
# MAGIC   "pymupdf==1.24.13" \
# MAGIC   "markdownify==0.12.1" \
# MAGIC   "transformers==4.41.1" \
# MAGIC   "tiktoken==0.7.0" \
# MAGIC   "langchain-text-splitters==0.2.0" \
# MAGIC   "pypandoc_binary==1.13" \
# MAGIC   "pyyaml"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫 ✏️ Define utility classes and functions
# MAGIC
# MAGIC Define utility functions. This is done to add modularization to the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serialization functions
# MAGIC The goal of serialization is to save the class name (e.g., `util.xx.xx.configClassName`) with the dumped YAML.
# MAGIC This allows ANY config to be dynamically loaded from a YAML without knowing about the `configClassName` before OR having it imported in your Python env.
# MAGIC
# MAGIC This is necessary for MultiAgent.`agents` and FunctionCallingAgent.`tools` since they can have multiple types of agent or tool configs in them -- when the config is loaded in the serving or local env, we don't know what these `configClassName` will be ahead of time & we want to avoid importing them all in the Python env.
# MAGIC
# MAGIC
# MAGIC #### How it works:
# MAGIC The ONLY way to dump a class is to call `model_dump()` on it, which will return a dict with the `_CLASS_PATH_KEY` key containing the class path e.g., `util.xx.xx.configClassName`.
# MAGIC
# MAGIC All other dumping methods (yaml, etc) call model_dump() since it is a Pydantic method. The ONLY way to load a serialized class is to call `load_obj_from_yaml` with the YAML string.
# MAGIC `load_obj_from_yaml` will parse the YAML string and get the class path key.
# MAGIC It will then use that class path key to dynamically load the class from the Python path.
# MAGIC It will then call that class's _load_class_from_dict method with the remaining data to let it do anything custom e.g,. load the tools or the agents.
# MAGIC
# MAGIC If you haven't overridden `_load_class_from_dict`, it will call the default implementation of this method from `SerializableModel`
# MAGIC otherwise, it will call your overridden `_load_class_from_dict` method.
# MAGIC
# MAGIC ### How to use:
# MAGIC Inherit your config class from `SerializableModel`.
# MAGIC
# MAGIC If you don't have any `SerializableModel` fields, you can call `load_obj_from_yaml` directly on your class's dumped YAML string; nothing else is required.
# MAGIC
# MAGIC If you have SerializableModel fields, you must:
# MAGIC 1. Override the _load_class_from_dict method to handle the deserialization of those sub-configs
# MAGIC 2. Override the model_dump method to call the model_dump of each of those sub-configs properly
# MAGIC
# MAGIC ### Examples
# MAGIC 1. No sub-configs: GenieAgentConfig, UCTool
# MAGIC 2. Has sub-configs: FunctionCallingAgentConfig (in `tools`), MultiAgentConfig (in `agents`)
# MAGIC load_obj_from_yaml --> The only way a class is loaded will get the classpath key
# MAGIC
# MAGIC TODO: add tests.  this was tested manually in a notebook verifying that all classes worked.

# COMMAND ----------

# DBTITLE 1,Define serialized config class and SDK helpers
from typing import Any, Dict, Tuple, Type
import yaml
from pydantic import BaseModel
import importlib
import json


def serializable_config_to_yaml(obj: BaseModel) -> str:
    data = obj.model_dump()
    return yaml.dump(data)

# TODO: add tests.  this was tested manually in a notebook verifying that all classes worked.


_CLASS_PATH_KEY = "class_path"


class SerializableConfig(BaseModel):
    def to_yaml(self) -> str:
        return serializable_config_to_yaml(self)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped[_CLASS_PATH_KEY] = f"{self.__module__}.{self.__class__.__name__}"
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        return class_object(**data)

    def pretty_print(self):
        print(json.dumps(self.model_dump(), indent=2))


def serializable_config_to_yaml_file(obj: BaseModel, yaml_file_path: str) -> None:
    with open(yaml_file_path, "w") as handle:
        handle.write(serializable_config_to_yaml(obj))


# Helper method used by SerializableModel's with fields containing SerializableModels
def _load_class_from_dict(data: Dict[str, Any]) -> Tuple[Type, Dict[str, Any]]:
    """Dynamically load a class from data containing a class path.

    Args:
        data: Dictionary containing _CLASS_PATH_KEY and other data

    Returns:
        Tuple[Type, Dict[str, Any]]: The class object and the remaining data
    """
    class_path = data.pop(_CLASS_PATH_KEY)

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name), data


def load_serializable_config_from_yaml(yaml_str: str) -> SerializableConfig:
    data = yaml.safe_load(yaml_str)# Helper functions for displaying Delta Table and Volume URLs

from typing import Optional
import json
import subprocess

from databricks.sdk import WorkspaceClient
from mlflow.utils import databricks_utils as du


def get_databricks_cli_config() -> dict:
    """Retrieve the Databricks CLI configuration by running 'databricks auth describe' command.

    Returns:
        dict: The parsed JSON configuration from the Databricks CLI, or None if an error occurs

    Note:
        Requires the Databricks CLI to be installed and configured
    """
    try:
        # Run databricks auth describe command and capture output
        process = subprocess.run(
            ["databricks", "auth", "describe", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError if command fails
        )

        # Parse JSON output
        return json.loads(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running databricks CLI command: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing databricks CLI JSON output: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting databricks config from CLI: {e}")
        return None


def get_workspace_hostname() -> str:
    """Get the Databricks workspace hostname.

    Returns:
        str: The full workspace hostname (e.g., 'https://my-workspace.cloud.databricks.com')

    Raises:
        RuntimeError: If not in a Databricks notebook and unable to get workspace hostname from CLI config
    """
    if du.is_in_databricks_notebook():
        return "https://" + du.get_browser_hostname()
    else:
        cli_config = get_databricks_cli_config()
        if cli_config is None:
            raise RuntimeError("Could not get Databricks CLI config")
        try:
            return cli_config["details"]["host"]
        except KeyError:
            raise RuntimeError(
                "Could not find workspace hostname in Databricks CLI config"
            )


def get_table_url(table_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog table in the Databricks UI.

    Args:
        table_fqdn: Fully qualified table name in format 'catalog.schema.table'.
                   Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the table in the Databricks UI.

    Example:
        >>> get_table_url("main.default.my_table")
        'https://my-workspace.cloud.databricks.com/explore/data/main/default/my_table'
    """
    table_fqdn = table_fqdn.replace("`", "")
    catalog, schema, table = table_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/{catalog}/{schema}/{table}"
    return url


def get_volume_url(volume_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog volume in the Databricks UI.

    Args:
        volume_fqdn: Fully qualified volume name in format 'catalog.schema.volume'.
                    Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the volume in the Databricks UI.

    Example:
        >>> get_volume_url("main.default.my_volume")
        'https://my-workspace.cloud.databricks.com/explore/data/volumes/main/default/my_volume'
    """
    volume_fqdn = volume_fqdn.replace("`", "")
    catalog, schema, volume = volume_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/volumes/{catalog}/{schema}/{volume}"
    return url


def get_mlflow_experiment_url(experiment_id: str) -> str:
    """Generate the URL for an MLflow experiment in the Databricks UI.

    Args:
        experiment_id: The ID of the MLflow experiment

    Returns:
        str: The full URL to view the MLflow experiment in the Databricks UI.

    Example:
        >>> get_mlflow_experiment_url("<experiment_id>")
        'https://my-workspace.cloud.databricks.com/ml/experiments/<experiment_id>'
    """
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/ml/experiments/{experiment_id}"
    return url


def get_mlflow_experiment_traces_url(experiment_id: str) -> str:
    """Generate the URL for the MLflow experiment traces in the Databricks UI."""
    return get_mlflow_experiment_url(experiment_id) + "?compareRunsMode=TRACES"


def get_function_url(function_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog function in the Databricks UI.

    Args:
        function_fqdn: Fully qualified function name in format 'catalog.schema.function'.
                      Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the function in the Databricks UI.

    Example:
        >>> get_function_url("main.default.my_function")
        'https://my-workspace.cloud.databricks.com/explore/data/functions/main/default/my_function'
    """
    function_fqdn = function_fqdn.replace("`", "")
    catalog, schema, function = function_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/functions/{catalog}/{schema}/{function}"
    return url


def get_cluster_url(cluster_id: str) -> str:
    """Generate the URL for a Databricks cluster in the Databricks UI.

    Args:
        cluster_id: The ID of the cluster

    Returns:
        str: The full URL to view the cluster in the Databricks UI.

    Example:
        >>> get_cluster_url("<cluster_id>")
        'https://my-workspace.cloud.databricks.com/compute/clusters/<cluster_id>'
    """
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/compute/clusters/{cluster_id}"
    return url


def get_active_cluster_id_from_databricks_auth() -> Optional[str]:
    """Get the active cluster ID from the Databricks CLI authentication configuration.

    Returns:
        Optional[str]: The active cluster ID if found, None if not found or if an error occurs

    Note:
        This function relies on the Databricks CLI configuration having a cluster_id set
    """
    if du.is_in_databricks_notebook():
        raise ValueError(
            "Cannot get active cluster ID from the Databricks CLI in a Databricks notebook"
        )
    try:
        # Get config from the databricks cli
        auth_output = get_databricks_cli_config()

        # Safely navigate nested dict
        details = auth_output.get("details", {})
        config = details.get("configuration", {})
        cluster = config.get("cluster_id", {})
        cluster_id = cluster.get("value")

        if cluster_id is None:
            raise ValueError("Could not find cluster_id in Databricks auth config")

        return cluster_id

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def get_active_cluster_id() -> Optional[str]:
    """Get the active cluster ID.

    Returns:
        Optional[str]: The active cluster ID if found, None if not found or if an error occurs
    """
    if du.is_in_databricks_notebook():
        return du.get_active_cluster_id()
    else:
        return get_active_cluster_id_from_databricks_auth()


def get_current_user_info(spark) -> tuple[str, str, str]:
    # Get current user's name & email
    w = WorkspaceClient()
    user_email = w.current_user.me().user_name
    user_name = user_email.split("@")[0].replace(".", "_")

    # Get the workspace default UC catalog
    default_catalog = spark.sql("select current_catalog() as cur_catalog").collect()[0][
        "cur_catalog"
    ]

    return user_email, user_name, default_catalog

    class_obj, remaining_data = _load_class_from_dict(data)
    return class_obj._load_class_from_dict(class_obj, remaining_data)


def load_serializable_config_from_yaml_file(yaml_file_path: str) -> SerializableConfig:
    with open(yaml_file_path, "r") as file:
        return load_serializable_config_from_yaml(file.read())


# COMMAND ----------

# DBTITLE 1,Define Unity Catalog volume source config class
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.catalog import VolumeType
from pydantic import Field, computed_field, field_validator


class UCVolumeSourceConfig(SerializableConfig):
    """
    Source data configuration for the Unstructured Data Pipeline. You can modify this class to add additional configuration settings.

    Args:
      uc_catalog_name (str):
        Required. Name of the Unity Catalog.
      uc_schema_name (str):
        Required. Name of the Unity Catalog schema.
      uc_volume_name (str):
        Required. Name of the Unity Catalog volume.
    """

    @field_validator("uc_catalog_name", "uc_schema_name", "uc_volume_name")
    def validate_not_default(cls, value: str) -> str:
        if value == "REPLACE_ME":
            raise ValueError(
                "Please replace the default value 'REPLACE_ME' with your actual configuration"
            )
        return value

    uc_catalog_name: str = Field(..., min_length=1)
    uc_schema_name: str = Field(..., min_length=1)
    uc_volume_name: str = Field(..., min_length=1)

    @computed_field()
    def volume_path(self) -> str:
        return f"/Volumes/{self.uc_catalog_name}/{self.uc_schema_name}/{self.uc_volume_name}"

    @computed_field()
    def volume_uc_fqn(self) -> str:
        return f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_volume_name}"

    def check_if_volume_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            # Use the computed field instead of reconstructing the FQN
            w.volumes.read(name=self.volume_uc_fqn)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def create_volume(self):
        try:
            w = WorkspaceClient()
            w.volumes.create(
                catalog_name=self.uc_catalog_name,
                schema_name=self.uc_schema_name,
                name=self.uc_volume_name,
                volume_type=VolumeType.MANAGED,
            )
        except ResourceAlreadyExists:
            pass

    def check_if_catalog_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            w.catalogs.get(name=self.uc_catalog_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def check_if_schema_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            full_name = f"{self.uc_catalog_name}.{self.uc_schema_name}"
            w.schemas.get(full_name=full_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def create_or_validate_volume(self) -> tuple[bool, str]:
        """
        Validates that the volume exists and creates it if it doesn't
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        if not self.check_if_catalog_exists():
            msg = f"Catalog '{self.uc_catalog_name}' does not exist. Please create it first."
            return (False, msg)

        if not self.check_if_schema_exists():
            msg = f"Schema '{self.uc_schema_name}' does not exist in catalog '{self.uc_catalog_name}'. Please create it first."
            return (False, msg)

        if not self.check_if_volume_exists():
            print(f"Volume {self.volume_path} does not exist. Creating...")
            try:
                self.create_volume()
            except Exception as e:
                msg = f"Failed to create volume: {str(e)}"
                return (False, msg)
            msg = f"Successfully created volume {self.volume_path}. View here: {get_volume_url(self.volume_uc_fqn)}"
            print(msg)
            return (True, msg)

        msg = f"Volume {self.volume_path} exists.  View here: {get_volume_url(self.volume_uc_fqn)}"
        print(msg)
        return (True, msg)

    def list_files(self) -> list[str]:
        """
        Lists all files in the Unity Catalog volume using dbutils.fs.

        Returns:
            list[str]: A list of file paths in the volume

        Raises:
            Exception: If the volume doesn't exist or there's an error accessing it
        """
        if not self.check_if_volume_exists():
            raise Exception(f"Volume {self.volume_path} does not exist")

        w = WorkspaceClient()
        try:
            # List contents using dbutils.fs
            files = w.dbutils.fs.ls(self.volume_path)
            return [file.name for file in files]
        except Exception as e:
            raise Exception(f"Failed to list files in volume: {str(e)}")


# COMMAND ----------

# DBTITLE 1,Define output config class and related helpers
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType


class DataPipelineOutputConfig(SerializableConfig):
    """Configuration for managing output locations and naming conventions in the data pipeline.

    This class handles the configuration of table names and vector search endpoints for the data pipeline.
    It follows a consistent naming pattern for all generated tables and provides version control capabilities.

    Naming Convention:
        {catalog}.{schema}.{base_table_name}_{table_postfix}__{version_suffix}

    Generated Tables:
        1. Parsed docs table: Stores the raw parsed documents
        2. Chunked docs table: Stores the documents split into chunks
        3. Vector index: Stores the vector embeddings for search

    Args:
        uc_catalog_name (str): Unity Catalog name where tables will be created
        uc_schema_name (str): Schema name within the catalog
        base_table_name (str): Core name used as prefix for all generated tables
        docs_table_postfix (str, optional): Suffix for the parsed documents table. Defaults to "docs"
        chunked_table_postfix (str, optional): Suffix for the chunked documents table. Defaults to "docs_chunked"
        vector_index_postfix (str, optional): Suffix for the vector index. Defaults to "docs_chunked_index"
        version_suffix (str, optional): Version identifier (e.g., 'v1', 'test') to maintain multiple pipeline versions
        vector_search_endpoint (str): Name of the vector search endpoint to use

    Examples:
        With version_suffix="v1":
            >>> config = DataPipelineOuputConfig(
            ...     uc_catalog_name="my_catalog",
            ...     uc_schema_name="my_schema",
            ...     base_table_name="agent",
            ...     version_suffix="v1"
            ... )
            # Generated tables:
            # - my_catalog.my_schema.agent_docs__v1
            # - my_catalog.my_schema.agent_docs_chunked__v1
            # - my_catalog.my_schema.agent_docs_chunked_index__v1

        Without version_suffix:
            # - my_catalog.my_schema.agent_docs
            # - my_catalog.my_schema.agent_docs_chunked
            # - my_catalog.my_schema.agent_docs_chunked_index
    """

    vector_search_endpoint: str
    parsed_docs_table: str
    chunked_docs_table: str
    vector_index: str

    def __init__(
        self,
        *,
        vector_search_endpoint: str,
        parsed_docs_table: Optional[str] = None,
        chunked_docs_table: Optional[str] = None,
        vector_index: Optional[str] = None,
        uc_catalog_name: Optional[str] = None,
        uc_schema_name: Optional[str] = None,
        base_table_name: Optional[str] = None,
        docs_table_postfix: str = "docs",
        chunked_table_postfix: str = "docs_chunked",
        vector_index_postfix: str = "docs_chunked_index",
        version_suffix: Optional[str] = None,
    ):
        """Initialize a new DataPipelineOuputConfig instance.

        Supports two initialization styles:
        1. Direct table names:
            - parsed_docs_table
            - chunked_docs_table
            - vector_index

        2. Generated table names using:
            - uc_catalog_name
            - uc_schema_name
            - base_table_name
            - [optional] postfixes and version_suffix

        Args:
            vector_search_endpoint (str): Name of the vector search endpoint to use
            parsed_docs_table (str, optional): Direct table name for parsed docs
            chunked_docs_table (str, optional): Direct table name for chunked docs
            vector_index (str, optional): Direct name for vector index
            uc_catalog_name (str, optional): Unity Catalog name where tables will be created
            uc_schema_name (str, optional): Schema name within the catalog
            base_table_name (str, optional): Core name used as prefix for all generated tables
            docs_table_postfix (str, optional): Suffix for parsed documents table. Defaults to "docs"
            chunked_table_postfix (str, optional): Suffix for chunked documents table. Defaults to "docs_chunked"
            vector_index_postfix (str, optional): Suffix for vector index. Defaults to "docs_chunked_index"
            version_suffix (str, optional): Version identifier for multiple pipeline versions
        """
        _validate_not_default(vector_search_endpoint)

        if parsed_docs_table and chunked_docs_table and vector_index:
            # Direct table names provided
            if any([uc_catalog_name, uc_schema_name, base_table_name]):
                raise ValueError(
                    "Cannot provide both direct table names and table name generation parameters"
                )
        elif all([uc_catalog_name, uc_schema_name, base_table_name]):
            # Generate table names
            _validate_not_default(uc_catalog_name)
            _validate_not_default(uc_schema_name)
            _validate_not_default(base_table_name)

            parsed_docs_table = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                docs_table_postfix,
                version_suffix,
            )
            chunked_docs_table = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                chunked_table_postfix,
                version_suffix,
            )
            vector_index = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                vector_index_postfix,
                version_suffix,
                escape=False,
            )
        else:
            raise ValueError(
                "Must provide either all direct table names or all table name generation parameters"
            )

        super().__init__(
            parsed_docs_table=parsed_docs_table,
            chunked_docs_table=chunked_docs_table,
            vector_index=vector_index,
            vector_search_endpoint=vector_search_endpoint,
        )

    def check_if_vector_search_endpoint_exists(self):
        w = WorkspaceClient()
        vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
        if (
            sum(
                [
                    self.vector_search_endpoint == ve.name
                    for ve in vector_search_endpoints
                ]
            )
            == 0
        ):
            return False
        else:
            return True

    def create_vector_search_endpoint(self):
        w = WorkspaceClient()
        print(
            f"Please wait, creating Vector Search endpoint `{self.vector_search_endpoint}`.  This can take up to 20 minutes..."
        )
        w.vector_search_endpoints.create_endpoint_and_wait(
            self.vector_search_endpoint, endpoint_type=EndpointType.STANDARD
        )
        # Make sure vector search endpoint is online and ready.
        w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            self.vector_search_endpoint
        )

    def create_or_validate_vector_search_endpoint(self):
        if not self.check_if_vector_search_endpoint_exists():
            self.create_vector_search_endpoint()
        return self.validate_vector_search_endpoint()

    def validate_vector_search_endpoint(self) -> tuple[bool, str]:
        """
        Validates that the specified Vector Search endpoint exists
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        if not self.check_if_vector_search_endpoint_exists():
            msg = f"Vector Search endpoint '{self.vector_search_endpoint}' does not exist. Please either manually create it or call `output_config.create_or_validate_vector_search_endpoint()` to create it."
            return (False, msg)

        msg = f"Vector Search endpoint '{self.vector_search_endpoint}' exists."
        print(msg)
        return (True, msg)

    def validate_catalog_and_schema(self) -> tuple[bool, str]:
        """
        Validates that the specified catalog and schema exist
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """

        # Check catalog and schema for parsed_docs_table
        parsed_docs_catalog = _get_uc_catalog_name(self.parsed_docs_table)
        parsed_docs_schema = _get_uc_schema_name(self.parsed_docs_table)
        if not _check_if_catalog_exists(parsed_docs_catalog):
            msg = f"Catalog '{parsed_docs_catalog}' does not exist for parsed_docs_table. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(parsed_docs_catalog, parsed_docs_schema):
            msg = f"Schema '{parsed_docs_schema}' does not exist in catalog '{parsed_docs_catalog}' for parsed_docs_table. Please create it first."
            return (False, msg)

        # Check catalog and schema for chunked_docs_table
        chunked_docs_catalog = _get_uc_catalog_name(self.chunked_docs_table)
        chunked_docs_schema = _get_uc_schema_name(self.chunked_docs_table)
        if not _check_if_catalog_exists(chunked_docs_catalog):
            msg = f"Catalog '{chunked_docs_catalog}' does not exist for chunked_docs_table. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(chunked_docs_catalog, chunked_docs_schema):
            msg = f"Schema '{chunked_docs_schema}' does not exist in catalog '{chunked_docs_catalog}' for chunked_docs_table. Please create it first."
            return (False, msg)

        # Check catalog and schema for vector_index
        vector_index_catalog = _get_uc_catalog_name(self.vector_index)
        vector_index_schema = _get_uc_schema_name(self.vector_index)
        if not _check_if_catalog_exists(vector_index_catalog):
            msg = f"Catalog '{vector_index_catalog}' does not exist for vector_index. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(vector_index_catalog, vector_index_schema):
            msg = f"Schema '{vector_index_schema}' does not exist in catalog '{vector_index_catalog}' for vector_index. Please create it first."
            return (False, msg)

        msg = f"All catalogs and schemas exist for parsed_docs_table, chunked_docs_table, and vector_index."
        print(msg)
        return (True, msg)


def _escape_uc_fqn(uc_fqn: str) -> str:
    """
    Escape the fully qualified name (FQN) for a Unity Catalog asset if it contains special characters.

    Args:
        uc_fqn (str): The fully qualified name of the asset.

    Returns:
        str: The escaped fully qualified name if it contains special characters, otherwise the original FQN.
    """
    if "-" in uc_fqn:
        parts = uc_fqn.split(".")
        escaped_parts = [f"`{part}`" for part in parts]
        return ".".join(escaped_parts)
    else:
        return uc_fqn


def _build_table_name(
    uc_catalog_name: str,
    uc_schema_name: str,
    base_table_name: str,
    postfix: str,
    version_suffix: str = None,
    escape: bool = True,
) -> str:
    """Helper to build consistent table names

    Args:
        postfix: The table name postfix to append
        escape: Whether to escape special characters in the table name. Defaults to True.

    Returns:
        The constructed table name, optionally escaped
    """
    suffix = f"__{version_suffix}" if version_suffix else ""
    raw_name = f"{uc_catalog_name}.{uc_schema_name}.{base_table_name}_{postfix}{suffix}"
    return _escape_uc_fqn(raw_name) if escape else raw_name


def _validate_not_default(value: str) -> str:
    if value == "REPLACE_ME":
        raise ValueError(
            "Please replace the default value 'REPLACE_ME' with your actual configuration"
        )
    return value


def _get_uc_catalog_name(uc_fqn: str) -> str:
    unescaped_uc_fqn = uc_fqn.replace("`", "")
    return unescaped_uc_fqn.split(".")[0]


def _get_uc_schema_name(uc_fqn: str) -> str:
    unescaped_uc_fqn = uc_fqn.replace("`", "")
    return unescaped_uc_fqn.split(".")[1]


def _check_if_catalog_exists(uc_catalog_name) -> bool:
    w = WorkspaceClient()
    try:
        w.catalogs.get(name=uc_catalog_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False


def _check_if_schema_exists(uc_catalog_name, uc_schema_name) -> bool:
    w = WorkspaceClient()
    try:
        full_name = f"{uc_catalog_name}.{uc_schema_name}"
        w.schemas.get(full_name=full_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False


# COMMAND ----------

import requests
import collections
import os


def download_file_from_git(dest, owner, repo, path):
    def download_file(url, destination):
        local_filename = url.split("/")[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            print("saving " + destination + "/" + local_filename)
            with open(destination + "/" + local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename

    if not os.path.exists(dest):
        os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor

    files = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents{path}"
    ).json()
    files = [f["download_url"] for f in files if "NOTICE" not in f["name"]]

    def download_to_dest(url):
        download_file(url, dest)

    with ThreadPoolExecutor(max_workers=10) as executor:
        collections.deque(executor.map(download_to_dest, files))

# COMMAND ----------

# MAGIC %md
# MAGIC # 📂 Data source & destination configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ✏️ Configure the data pipeline's source location.
# MAGIC
# MAGIC Choose a [Unity Catalog Volume](https://docs.databricks.com/en/volumes/index.html) containing PDF, HTML, etc... documents to be parsed, chunked, and embedded.
# MAGIC
# MAGIC Use the widgets at the top of the notebook to choose the following values:
# MAGIC
# MAGIC - `uc_catalog_name`: Name of the Unity Catalog.
# MAGIC - `uc_schema_name`: Name of the Unity Catalog schema.
# MAGIC - `uc_volume_name`: Name of the Unity Catalog volume.
# MAGIC
# MAGIC Running these cells will validate that the Unity Catalog Volume exists and try to create it if it does not.
# MAGIC
# MAGIC The code in this section is organized around a class to represent the Unity Catalog Volume as a source for your data pipeline and an associated parent class for managing serializable configuration objects. The primary cell to focus on is the one that configures and validates the source object, **Configure and create or validate the volume**.
# MAGIC

# COMMAND ----------


dbutils.widgets.text("db_name",'',label="Database")
dbutils.widgets.text("catalog", '',label="Catalog")
dbutils.widgets.text("volume_name", '',label="Volume Name")

uc_catalog_name = dbutils.widgets.get("catalog")
uc_schema_name = dbutils.widgets.get("db_name")
uc_volume_name = dbutils.widgets.get("volume_name")

if not uc_catalog_name or not uc_schema_name  or not uc_volume_name:
  print("Please set all the Data Configurations")



# Configure the Unity Catalog Volume that contains the source documents
source_config = UCVolumeSourceConfig(
  uc_catalog_name = dbutils.widgets.get("catalog"),
  uc_schema_name = dbutils.widgets.get("db_name"),
  uc_volume_name = dbutils.widgets.get("volume_name")
)

# Check if volume exists, create otherwise
is_valid, msg = source_config.create_or_validate_volume()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# DBTITLE 1,Get the pdfs from a remote location
volume_path =  f'/Volumes/{uc_catalog_name}/{uc_schema_name}/{uc_volume_name}'

owner = "databricks-demos"
repo = "dbdemos-dataset"
path =  "/llm/databricks-pdf-documentation"
files = dbutils.fs.ls(volume_path)

if not files:
    download_file_from_git(volume_path, owner, repo, path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ✏️ Configure the data pipeline's output location.
# MAGIC  
# MAGIC Choose where the data pipeline outputs the parsed, chunked, and embedded documents.
# MAGIC
# MAGIC Required parameters:
# MAGIC * `uc_catalog_name`: Unity Catalog name where tables will be created
# MAGIC * `uc_schema_name`: Schema name in the catalog 
# MAGIC * `base_table_name`: Core name used as a prefix for all generated tables
# MAGIC * `vector_search_endpoint`: Vector Search endpoint to store the index
# MAGIC
# MAGIC Optional parameters:
# MAGIC * `docs_table_postfix`: Suffix for the parsed documents table (default: "docs")
# MAGIC * `chunked_table_postfix`: Suffix for the chunked documents table (default: "docs_chunked") 
# MAGIC * `vector_index_postfix`: Suffix for the vector index (default: "docs_chunked_index")
# MAGIC * `version_suffix`: Version identifier (e.g., 'v1', 'test') to maintain multiple versions
# MAGIC
# MAGIC The generated tables follow this naming convention:
# MAGIC * Parsed docs: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{docs_table_postfix}__{version_suffix}
# MAGIC * Chunked docs: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{chunked_table_postfix}__{version_suffix}
# MAGIC * Vector index: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{vector_index_postfix}__{version_suffix}
# MAGIC
# MAGIC *Note: If you are comparing different chunking/parsing/embedding strategies, set the `version_suffix` parameter to maintain multiple versions of the pipeline output with the same base_table_name.*
# MAGIC
# MAGIC *Databricks suggests sharing a Vector Search endpoint across multiple agents.*

# COMMAND ----------

# DBTITLE 1,Create and validate the output config
# Output configuration
output_config = DataPipelineOutputConfig(
    # Required parameters
    uc_catalog_name=source_config.uc_catalog_name, # usually same as source volume catalog, by default is the same as the source volume catalog
    uc_schema_name=source_config.uc_schema_name, # usually same as source volume schema, by default is the same as the source volume schema
    base_table_name=source_config.uc_volume_name, # usually similar / same as the source volume name; by default, is the same as the volume_name
    # vector_search_endpoint="REPLACE_ME", # Vector Search endpoint to store the index
    vector_search_endpoint="aliciachimeno_ext_vector_search", # Vector Search endpoint to store the index

    # Optional parameters, showing defaults
    docs_table_postfix="docs",              # default value is `docs`
    chunked_table_postfix="docs_chunked",   # default value is `docs_chunked`
    vector_index_postfix="docs_chunked_index", # default value is `docs_chunked_index`
    version_suffix= None                     # default is None

    # Output tables / indexes follow this naming convention:
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{docs_table_postfix}__{version_suffix}
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{chunked_table_postfix}__{version_suffix}
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{vector_index_postfix}__{version_suffix}
)

# Alternatively, you can directly pass in the Unity Catalog locations of the tables / indexes
# output_config = DataPipelineOutputConfig(
#     chunked_docs_table="catalog.schema.docs_chunked",
#     parsed_docs_table="catalog.schema.parsed_docs",
#     vector_index="catalog.schema.docs_chunked_index",
#     vector_search_endpoint="REPLACE_ME",
# )

# Check Unity Catalog locations exists
is_valid, msg = output_config.validate_catalog_and_schema()
if not is_valid:
    raise Exception(msg)

# Check Vector Search endpoint exists
is_valid, msg = output_config.create_or_validate_vector_search_endpoint()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ✏️ Configure chunk size and embedding model
# MAGIC
# MAGIC **Chunk size and overlap** control how a larger document is turned into smaller chunks that an embedding model can process.  See Databricks documentation - Chunking for more information ([AWS](https://docs.databricks.com/aws/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag#chunking) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag#chunking))
# MAGIC
# MAGIC **The embedding model** is an AI model that identifies the most similar documents to a user's query.  See Databricks documentation - Embedding model for more details ([AWS](https://docs.databricks.com/aws/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag#embedding-model) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag#embedding-model)).
# MAGIC
# MAGIC This notebook supports the following [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/embeddings`/.  If you want to try another model, you must modify `utils/get_recursive_character_text_splitter` to add support.
# MAGIC - `databricks-gte-large-en` or `databricks-bge-large-en`
# MAGIC - Azure OpenAI or OpenAI External Model of type `text-embedding-ada-002`, `text-embedding-3-small` or `text-embedding-3-large`

# COMMAND ----------

# DBTITLE 1,Define chunking and embedding helpers
from typing import Callable, Tuple, Optional
from databricks.sdk import WorkspaceClient
from pydantic import BaseModel

# Constants
HF_CACHE_DIR = "/local_disk0/tmp/hf_cache/"

# Embedding Models Configuration
EMBEDDING_MODELS = {
    "gte-large-en-v1.5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "Alibaba-NLP/gte-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 8192,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge-large-en-v1.5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge_large_en_v1_5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "text-embedding-ada-002": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-ada-002"),
        "type": "OPENAI",
    },
    "text-embedding-3-small": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-small"),
        "type": "OPENAI",
    },
    "text-embedding-3-large": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-large"),
        "type": "OPENAI",
    },
}


def get_workspace_client() -> WorkspaceClient:
    """Returns a WorkspaceClient instance."""
    return WorkspaceClient()


# TODO: this is a cheap hack to avoid importing tokenizer libs at the top level -  the datapipeline utils are imported by the agent notebook which won't have these libs loaded & we don't want to since autotokenizer is heavy weight.
def get_embedding_model_tokenizer(endpoint_type: str) -> Optional[dict]:
    from transformers import AutoTokenizer
    import tiktoken

    # copy here to prevent needing to install tokenizer libraries everywhere this is imported
    EMBEDDING_MODELS_W_TOKENIZER = {
        "gte-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "Alibaba-NLP/gte-large-en-v1.5", cache_dir=HF_CACHE_DIR
            ),
            "context_window": 8192,
            "type": "SENTENCE_TRANSFORMER",
        },
        "bge-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
            ),
            "context_window": 512,
            "type": "SENTENCE_TRANSFORMER",
        },
        "bge_large_en_v1_5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
            ),
            "context_window": 512,
            "type": "SENTENCE_TRANSFORMER",
        },
        "text-embedding-ada-002": {
            "context_window": 8192,
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-ada-002"),
            "type": "OPENAI",
        },
        "text-embedding-3-small": {
            "context_window": 8192,
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-small"),
            "type": "OPENAI",
        },
        "text-embedding-3-large": {
            "context_window": 8192,
            "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-large"),
            "type": "OPENAI",
        },
    }
    return EMBEDDING_MODELS_W_TOKENIZER.get(endpoint_type).get("tokenizer")


def get_embedding_model_config(endpoint_type: str) -> Optional[dict]:
    """
    Retrieve embedding model configuration by endpoint type.
    """

    return EMBEDDING_MODELS.get(endpoint_type)


def extract_endpoint_type(llm_endpoint) -> Optional[str]:
    """
    Extract the endpoint type from the given llm_endpoint object.
    """
    try:
        return llm_endpoint.config.served_entities[0].external_model.name
    except AttributeError:
        try:
            return llm_endpoint.config.served_entities[0].foundation_model.name
        except AttributeError:
            return None


def detect_fmapi_embedding_model_type(
    model_serving_endpoint: str,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Detects the embedding model type and configuration for the given endpoint.
    Returns a tuple of (endpoint_type, embedding_config) or (None, None) if not found.
    """
    client = get_workspace_client()

    try:
        llm_endpoint = client.serving_endpoints.get(name=model_serving_endpoint)
        endpoint_type = extract_endpoint_type(llm_endpoint)
    except Exception as e:
        endpoint_type = None

    embedding_config = (
        get_embedding_model_config(endpoint_type) if endpoint_type else None
    )

    embedding_config["tokenizer"] = (
        get_embedding_model_tokenizer(endpoint_type) if endpoint_type else None
    )

    return (endpoint_type, embedding_config)


def validate_chunk_size(chunk_spec: dict):
    """
    Validate the chunk size and overlap settings in chunk_spec.
    Raises ValueError if any condition is violated.
    """
    if (
        chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]
    ) > chunk_spec["context_window"]:
        msg = (
            f'Proposed chunk_size of {chunk_spec["chunk_size_tokens"]} + overlap of {chunk_spec["chunk_overlap_tokens"]} '
            f'is {chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]} which is greater than context '
            f'window of {chunk_spec["context_window"]} tokens.',
        )
        return (False, msg)
    elif chunk_spec["chunk_overlap_tokens"] > chunk_spec["chunk_size_tokens"]:
        msg = (
            f'Proposed `chunk_overlap_tokens` of {chunk_spec["chunk_overlap_tokens"]} is greater than the '
            f'`chunk_size_tokens` of {chunk_spec["chunk_size_tokens"]}. Reduce the size of `chunk_size_tokens`.',
        )
        return (False, msg)
    else:
        context_usage = (
            round(
                (chunk_spec["chunk_size_tokens"] + chunk_spec["chunk_overlap_tokens"])
                / chunk_spec["context_window"],
                2,
            )
            * 100
        )
        msg = f'Chunk size in tokens: {chunk_spec["chunk_size_tokens"]} and chunk overlap in tokens: {chunk_spec["chunk_overlap_tokens"]} are valid.  Using {round(context_usage, 2)}% ({chunk_spec["chunk_size_tokens"] + chunk_spec["chunk_overlap_tokens"]} tokens) of the {chunk_spec["context_window"]} token context window.'
        return (True, msg)

def get_recursive_character_text_splitter(
    model_serving_endpoint: str,
    embedding_model_name: str = None,
    chunk_size_tokens: int = None,
    chunk_overlap_tokens: int = 0,
) -> Callable[[str], list[str]]:
    """
    Creates a new function that, given an embedding endpoint, returns a callable that can chunk text documents. This utility allows you to write the core business logic of the chunker, without dealing with the details of text splitting. You can decide to write your own, or edit this code if it does not fit your use case.

    Args:
        model_serving_endpoint (str):
            The name of the Model Serving endpoint with the embedding model.
        embedding_model_name (str):
            The name of the embedding model e.g., `gte-large-en-v1.5`, etc.   If `model_serving_endpoint` is an OpenAI External Model or FMAPI model and set to `None`, this will be automatically detected.
        chunk_size_tokens (int):
            An optional size for each chunk in tokens. Defaults to `None`, which uses the model's entire context window.
        chunk_overlap_token (int):
            Tokens that should overlap between chunks. Defaults to `0`.

    Returns:
        A callable that takes a document (`str`) and produces a list of chunks (`list[str]`).
    """
    
    # imports here to prevent needing to install everywhere

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    import tiktoken

    try:
        # Detect the embedding model and its configuration
        embedding_model_name, chunk_spec = detect_fmapi_embedding_model_type(
            model_serving_endpoint
        )

        if chunk_spec is None or embedding_model_name is None:
            # Fall back to using provided embedding_model_name
            chunk_spec = EMBEDDING_MODELS.get(embedding_model_name)
            if chunk_spec is None:
                raise KeyError

        # Update chunk specification based on provided parameters
        chunk_spec["chunk_size_tokens"] = (
            chunk_size_tokens or chunk_spec["context_window"]
        )
        chunk_spec["chunk_overlap_tokens"] = chunk_overlap_tokens

        # Validate chunk size and overlap
        is_valid, msg = validate_chunk_size(chunk_spec)
        if not is_valid:
            raise ValueError(msg)
        else:
            print(msg)

    except KeyError:
        raise ValueError(
            f"Embedding model `{embedding_model_name}` not found. Available models: {EMBEDDING_MODELS.keys()}"
        )

    def _recursive_character_text_splitter(text: str) -> list[str]:
        tokenizer = chunk_spec["tokenizer"]()
        if chunk_spec["type"] == "SENTENCE_TRANSFORMER":
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
            )
        elif chunk_spec["type"] == "OPENAI":
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer.name,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
            )
        else:
            raise ValueError(f"Unsupported model type: {chunk_spec['type']}")
        return splitter.split_text(text)

    return _recursive_character_text_splitter

import re
from typing import Callable, List


import re
from typing import Callable

import re
from typing import Callable

def get_keyword_based_text_splitter(
    keyword: str,
    model_serving_endpoint: str = None,
    embedding_model_name: str = None,
) -> Callable[[str], list[str]]:
    """
    Creates a function that splits text based on a specific keyword, removing the keyword from the output.

    Args:
        keyword (str):
            The keyword used to split the text.
        model_serving_endpoint (str, optional):
            The name of the Model Serving endpoint (not used in this version, but kept for compatibility).
        embedding_model_name (str, optional):
            The name of the embedding model (not used in this version, but kept for compatibility).

    Returns:
        A callable that takes a document (`str`) and produces a list of chunks (`list[str]`).
    """
    if not keyword:
        raise ValueError("Keyword must be provided for splitting.")

    def _keyword_based_text_splitter(text: str) -> list[str]:
        chunks = re.split(rf'(?i){re.escape(keyword)}', text)
        return [chunk.strip() for chunk in chunks if chunk]
    
    return _keyword_based_text_splitter


# COMMAND ----------

# DBTITLE 1,Define chunking config class
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.serving import EndpointStateReady


class RecursiveTextSplitterChunkingConfig(SerializableConfig):
    """
    Configuration for the Unstructured Data Pipeline.

    Args:
        embedding_model_endpoint (str):
            Embedding model endpoint hosted on Model Serving.  Default is `databricks-gte-large`.  This can be an External Model, such as OpenAI, or a Databricks-hosted model on Foundational Model API. The list of Databricks-hosted models can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        chunk_size_tokens (int):
            The size of each chunk of the document in tokens. Default is 1024.
        chunk_overlap_tokens (int):
            The overlap of tokens between chunks. Default is 256.
    """

    embedding_model_endpoint: str = "databricks-gte-large-en"
    chunk_size_tokens: int = 1024
    chunk_overlap_tokens: int = 256

    def validate_embedding_endpoint(self) -> tuple[bool, str]:
        """
        Validates that the specified embedding endpoint exists and is of the correct type
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        task_type = "llm/v1/embeddings"
        w = WorkspaceClient()
        browser_url = get_workspace_hostname()
        try:
            llm_endpoint = w.serving_endpoints.get(name=self.embedding_model_endpoint)
        except ResourceDoesNotExist as e:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} not found."
            return (False, msg)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} is not in a READY state.  Please visit the status page to debug: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
            return (False, msg)
        if llm_endpoint.task != task_type:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} is online & ready, but does not support task type {task_type}.  Details at: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
            return (False, msg)

        msg = f"Validated serving endpoint {self.embedding_model_endpoint} as READY and of type {task_type}.  View here: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
        print(msg)
        return (True, msg)

    def validate_chunk_size_and_overlap(self) -> tuple[bool, str]:
        """
        Validates that chunk_size and overlap values are valid
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        # Detect the embedding model and its configuration
        embedding_model_name, chunk_spec = detect_fmapi_embedding_model_type(
            self.embedding_model_endpoint
        )

        # Update chunk specification based on provided parameters
        chunk_spec["chunk_size_tokens"] = self.chunk_size_tokens
        chunk_spec["chunk_overlap_tokens"] = self.chunk_overlap_tokens

        if chunk_spec is None or embedding_model_name is None:
            # Fall back to using provided embedding_model_name
            chunk_spec = EMBEDDING_MODELS.get(embedding_model_name)
            if chunk_spec is None:
                msg = f"Embedding model `{embedding_model_name}` not found, so can't validate chunking config. Chunking config must be validated for a specific embedding model.  Available models: {EMBEDDING_MODELS.keys()}"
                return (False, msg)

        # Validate chunk size and overlap
        is_valid, msg = validate_chunk_size(chunk_spec)
        if not is_valid:
            return (False, msg)
        else:
            print(msg)
            return (True, msg)


# COMMAND ----------

# DBTITLE 1,Configure and validate a chunking config instance
chunking_config = RecursiveTextSplitterChunkingConfig(
    embedding_model_endpoint="databricks-gte-large-en",  # A Model Serving endpoint supporting the /llm/v1/embeddings task
    chunk_size_tokens=1024,
    chunk_overlap_tokens=256,
)

# Validate the embedding endpoint & chunking config
is_valid, msg = chunking_config.validate_embedding_endpoint()
if not is_valid:
    raise Exception(msg)

is_valid, msg = chunking_config.validate_chunk_size_and_overlap()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫 ✏️ Write the data pipeline configuration to a YAML
# MAGIC
# MAGIC The following cells define a consolidated configuration object and write out an instance of it to a file so that it can be reloaded later by other components. For instance, this allows the configuration to be loaded and referenced by the Agent's notebook. You would want to move this class definition to a separate Python file in your code path and the refer to the same module by both your data pipeline and your agent, as demonstrated in the [GenAI cookbook](https://github.com/databricks/genai-cookbook/tree/main/openai_sdk_agent_app_sample_code). We include the class inline here simply for ease of use in having a single notebook to show an end-to-end pipeline for learning purposes.

# COMMAND ----------

# DBTITLE 1,Define overall pipeline config
from typing import Any, Dict


class DataPipelineConfig(SerializableConfig):
    source: UCVolumeSourceConfig
    output: DataPipelineOutputConfig
    chunking_config: RecursiveTextSplitterChunkingConfig

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["source"] = yaml.safe_load(
            serializable_config_to_yaml(self.source)
        )
        model_dumped["output"] = yaml.safe_load(
            serializable_config_to_yaml(self.output)
        )
        model_dumped["chunking_config"] = yaml.safe_load(
            serializable_config_to_yaml(self.chunking_config)
        )
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize sub-configs
        data["source"] = load_serializable_config_from_yaml(yaml.dump(data["source"]))
        data["output"] = load_serializable_config_from_yaml(yaml.dump(data["output"]))
        data["chunking_config"] = load_serializable_config_from_yaml(
            yaml.dump(data["chunking_config"])
        )
        return class_object(**data)


# COMMAND ----------

# DBTITLE 1,Create and save an instance of the pipeline config
data_pipeline_config = DataPipelineConfig(
    source=source_config,
    output=output_config,
    chunking_config=chunking_config,
)

serializable_config_to_yaml_file(data_pipeline_config, "./data_pipeline_config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛑 Pause - end of config section
# MAGIC
# MAGIC If you are running your initial data pipeline, you do not need to configure anything else, you can just `Run All` the notebook cells before.  You can modify these cells later to tune the quality of your data pipeline by changing the parsing logic.

# COMMAND ----------

# MAGIC %md
# MAGIC # ⌨️ Data pipeline code
# MAGIC
# MAGIC The code below executes the data pipeline.  You can modify the below code as indicated to implement different parsing or chunking strategies or to extract additional metadata fields

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ✏️ Step 1: Load and parse documents into a Delta table
# MAGIC
# MAGIC In this step, we'll load files from the Unity Catalog Volume defined in `source_config` into the Delta table `storage_config.parsed_docs_table` . The contents of each file will become a separate row in our Delta table.
# MAGIC
# MAGIC The path to the source document will be used as the `doc_uri`, which is displayed to your end users in the Agent Evaluation web application.
# MAGIC
# MAGIC After you evaluate the outputs and test your POC with stakeholders, you can return here to change the parsing logic or extraction.
# MAGIC
# MAGIC **Customize the parsing function**
# MAGIC
# MAGIC This default implementation parses PDF, HTML, and DOCX files using open source libraries. The first cells below define the parsing logic and its return value. If needed after your initial evaluation, Databricks suggest modifying the parsing logic to add support for more file types or extracting additional metadata about each document.

# COMMAND ----------

# DBTITLE 1,Define the file parsing logic
from typing import TypedDict
from datetime import datetime
import warnings
import traceback
import os
from urllib.parse import urlparse

# PDF libraries
import fitz
import pymupdf4llm

# HTML libraries
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

## JSON libraries
import json


# Schema of the dict returned by `file_parser(...)`
# This is used to create the output Delta Table's schema.
# Adjust the class if you want to add additional columns from your parser, such as extracting custom metadata.
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES
    # Parsed content of the document
    content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name

    # OK TO CHANGE THESE NAMES
    # Optionally, you can add additional metadata fields here
    # example_metadata: str
    last_modified: datetime


# Parser function.  Adjust this function to modify the parsing logic.
def file_parser(
    raw_doc_contents_bytes: bytes,
    doc_path: str,
    modification_time: datetime,
    doc_bytes_length: int,
) -> ParserReturnValue:
    """
    Parses the content of a PDF document into a string.

    This function takes the raw bytes of a PDF document and its path, attempts to parse the document using PyPDF,
    and returns the parsed content and the status of the parsing operation.

    Parameters:
    - raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed (set by Spark when loading the file)
    - doc_path (str): The DBFS path of the document, used to verify the file extension (set by Spark when loading the file)
    - modification_time (timestamp): The last modification time of the document (set by Spark when loading the file)
    - doc_bytes_length (long): The size of the document in bytes (set by Spark when loading the file)

    Returns:
    - ParserReturnValue: A dictionary containing the parsed document content and the status of the parsing operation.
      The 'contenty will contain the parsed text as a string, and the 'parser_status' key will indicate
      whether the parsing was successful or if an error occurred.
    """
    try:
        from markdownify import markdownify as md

        filename, file_extension = os.path.splitext(doc_path)

        if file_extension == ".pdf":
            pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(pdf_doc)

            parsed_document = {
                "content": md_text.strip(),
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".html":
            html_content = raw_doc_contents_bytes.decode("utf-8")

            markdown_contents = md(
                str(html_content).strip(), heading_style=markdownify.ATX
            )
            markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

            parsed_document = {
                "content": markdown_stripped,
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".docx":
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(raw_doc_contents_bytes)
                temp_file_path = temp_file.name
                md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

                parsed_document = {
                    "content": md.strip(),
                    "parser_status": "SUCCESS",
                }
        elif file_extension in [".txt", ".md"]:
            parsed_document = {
                "content": raw_doc_contents_bytes.decode("utf-8").strip(),
                "parser_status": "SUCCESS",
            }
        elif file_extension in [".json", ".jsonl"]:
            # NOTE: This is a placeholder for a JSON parser.  It's not a "real" parser, it just returns the raw JSON formatted into XML-like strings that LLMs tend to like.
            json_data = json.loads(raw_doc_contents_bytes.decode("utf-8"))

            def flatten_json_to_xml(obj, parent_key=""):
                xml_parts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            xml_parts.append(flatten_json_to_xml(value, key))
                        else:
                            xml_parts.append(f"<{key}>{str(value)}</{key}>")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            xml_parts.append(
                                flatten_json_to_xml(item, f"{parent_key}_{i}")
                            )
                        else:
                            xml_parts.append(
                                f"<{parent_key}_{i}>{str(item)}</{parent_key}_{i}>"
                            )
                else:
                    xml_parts.append(f"<{parent_key}>{str(obj)}</{parent_key}>")
                return "\n".join(xml_parts)

            flattened_content = flatten_json_to_xml(json_data)
            parsed_document = {
                "content": flattened_content.strip(),
                "parser_status": "SUCCESS",
            }
        else:
            raise Exception(f"No supported parser for {doc_path}")

        # Extract the required doc_uri
        # convert from `dbfs:/Volumes/catalog/schema/pdf_docs/filename.pdf` to `/Volumes/catalog/schema/pdf_docs/filename.pdf`
        modified_path = urlparse(doc_path).path
        parsed_document["doc_uri"] = modified_path

        # Sample metadata extraction logic
        # if "test" in parsed_document["content
        #     parsed_document["example_metadata"] = "test"
        # else:
        #     parsed_document["example_metadata"] = "not test"

        # Add the modified time
        parsed_document["last_modified"] = modification_time

        return parsed_document

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "content": "",
            "parser_status": f"ERROR: {status}",
        }


# COMMAND ----------

# DBTITLE 1,Define logic to apply the parsing function
import traceback
from datetime import datetime
from typing import Any, Callable, TypedDict, Dict
import os
from IPython.display import display_markdown
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType
from pyspark.sql import DataFrame, SparkSession


def _parse_and_extract(
    raw_doc_contents_bytes: bytes,
    modification_time: datetime,
    doc_bytes_length: int,
    doc_path: str,
    parse_file_udf: Callable[[[dict, Any]], str],
) -> Dict[str, Any]:
    """Parses raw bytes & extract metadata."""
    try:
        # Run the parser
        parser_output_dict = parse_file_udf(
            raw_doc_contents_bytes=raw_doc_contents_bytes,
            doc_path=doc_path,
            modification_time=modification_time,
            doc_bytes_length=doc_bytes_length,
        )

        if parser_output_dict.get("parser_status") == "SUCCESS":
            return parser_output_dict
        else:
            raise Exception(parser_output_dict.get("parser_status"))

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "content": "",
            "doc_uri": doc_path,
            "parser_status": status,
        }


def _get_parser_udf(
    # extract_metadata_udf: Callable[[[dict, Any]], str],
    parse_file_udf: Callable[[[dict, Any]], str],
    spark_dataframe_schema: StructType,
):
    """Gets the Spark UDF which will parse the files in parallel.

    Arguments:
      - extract_metadata_udf: A function that takes parsed content and extracts the metadata
      - parse_file_udf: A function that takes the raw file and returns the parsed text.
      - spark_dataframe_schema: The resulting schema of the document delta table
    """
    # This UDF will load each file, parse the doc, and extract metadata.
    parser_udf = func.udf(
        lambda raw_doc_contents_bytes, modification_time, doc_bytes_length, doc_path: _parse_and_extract(
            raw_doc_contents_bytes,
            modification_time,
            doc_bytes_length,
            doc_path,
            parse_file_udf,
        ),
        returnType=spark_dataframe_schema,
        useArrow=True,
    )
    return parser_udf


def load_files_to_df(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Load files from a directory into a Spark DataFrame.
    Each row in the DataFrame will contain the path, length, and content of the file; for more
    details, see https://spark.apache.org/docs/latest/sql-data-sources-binaryFile.html
    """

    print(f"Loading the raw files from {source_path}...")
    # Load the raw riles
    raw_files_df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(source_path)
    )

    # Check that files were present and loaded
    if raw_files_df.count() == 0:
        raise Exception(f"`{source_path}` does not contain any files.")

    # display_markdown(
    #     f"### Found {raw_files_df.count()} files in {source_path}: ", raw=True
    # )
    # raw_files_df.display()
    return raw_files_df


def apply_parsing_fn(
    raw_files_df: DataFrame,
    parse_file_fn: Callable[[[dict, Any]], str],
    parsed_df_schema: StructType,
) -> DataFrame:
    """
    Apply a file-parsing UDF to a DataFrame whose rows correspond to file content/metadata loaded via
    https://spark.apache.org/docs/latest/sql-data-sources-binaryFile.html
    Returns a DataFrame with the parsed content and metadata.
    """
    print(
        f"Applying parsing & metadata extraction to {raw_files_df.count()} files using Spark - this may take a long time if you have many documents..."
    )

    parser_udf = _get_parser_udf(parse_file_fn, parsed_df_schema)

    # Run the parsing
    parsed_files_staging_df = raw_files_df.withColumn(
        "parsing", parser_udf("content", "modificationTime", "length", "path")
    ).drop("content")

    # Filter for successfully parsed files
    parsed_files_df = parsed_files_staging_df  # .filter(
    #    parsed_files_staging_df.parsing.parser_status == "SUCCESS"
    # )

    # Change the schema to the resulting schema
    resulting_fields = [field.name for field in parsed_df_schema.fields]

    parsed_files_df = parsed_files_df.select(
        *[func.col(f"parsing.{field}").alias(field) for field in resulting_fields]
    )
    return parsed_files_df


# COMMAND ----------

# MAGIC %md
# MAGIC The cell below contains debugging code to test your parsing function on a single record. This is a good place to iterate as you adjust the parsing logic above to see how your changes impact the parser output.

# COMMAND ----------

# DBTITLE 1,Test the parsing logic on a few records
from pyspark.sql import functions as F

raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_config.volume_path,
)
print(f"Loaded {raw_files_df.count()} files from {source_config.volume_path}.  Files: {source_config.list_files()}")

test_records_dict = raw_files_df.toPandas().to_dict(orient="records")

for record in test_records_dict:
    print()
    print("Testing parsing for file: ", record["path"])
    print()
    test_result = file_parser(raw_doc_contents_bytes=record['content'], doc_path=record['path'], modification_time=record['modificationTime'], doc_bytes_length=record['length'])
    print(test_result)
    break # pause after 1 file.  if you want to test more files, remove the break statement


# COMMAND ----------

# MAGIC %md
# MAGIC 🚫✏️ The below cell is boilerplate code to apply the parsing function using Spark.

# COMMAND ----------

# DBTITLE 1,Define some helpers for creating the DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    ArrayType,
    TimestampType,
    DateType,
)
from typing import TypedDict, get_type_hints, List
from datetime import datetime, date, time


def typed_dict_to_spark_fields(typed_dict: type[TypedDict]) -> StructType:
    """
    Converts a TypedDict into a list of Spark StructField objects.

    This function maps Python types defined in a TypedDict to their corresponding
    Spark SQL data types, facilitating the creation of a Spark DataFrame schema
    from Python type annotations.

    Parameters:
    - typed_dict (type[TypedDict]): The TypedDict class to be converted.

    Returns:
    - StructType: A list of StructField objects representing the Spark schema.

    Raises:
    - ValueError: If an unsupported type is encountered or if dictionary types are used.
    """

    # Mapping of type names to Spark type objects
    type_mapping = {
        str: StringType(),
        int: IntegerType(),
        float: DoubleType(),
        bool: BooleanType(),
        list: ArrayType(StringType()),  # Default to StringType for arrays
        datetime: TimestampType(),
        date: DateType(),
    }

    def get_spark_type(value_type):
        """
        Helper function to map a Python type to a Spark SQL data type.

        This function supports basic Python types, lists of a single type, and raises
        an error for unsupported types or dictionaries.

        Parameters:
        - value_type: The Python type to be converted.

        Returns:
        - DataType: The corresponding Spark SQL data type.

        Raises:
        - ValueError: If the type is unsupported or if dictionary types are used.
        """
        if value_type in type_mapping:
            return type_mapping[value_type]
        elif hasattr(value_type, "__origin__") and value_type.__origin__ == list:
            # Handle List[type] types
            return ArrayType(get_spark_type(value_type.__args__[0]))
        elif hasattr(value_type, "__origin__") and value_type.__origin__ == dict:
            # Handle Dict[type, type] types (not fully supported)
            raise ValueError("Dict types are not fully supported")
        else:
            raise ValueError(f"Unsupported type: {value_type}")

    # Get the type hints for the TypedDict
    type_hints = get_type_hints(typed_dict)

    # Convert the type hints into a list of StructField objects
    fields = [
        StructField(key, get_spark_type(value), True)
        for key, value in type_hints.items()
    ]

    # Create and return the StructType object
    return fields


def typed_dicts_to_spark_schema(*typed_dicts: type[TypedDict]) -> StructType:
    """
    Converts multiple TypedDicts into a Spark schema.

    This function allows for the combination of multiple TypedDicts into a single
    Spark DataFrame schema, enabling the creation of complex data structures.

    Parameters:
    - *typed_dicts: Variable number of TypedDict classes to be converted.

    Returns:
    - StructType: A Spark schema represented as a StructType object, which is a collection
      of StructField objects derived from the provided TypedDicts.
    """
    fields = []
    for typed_dict in typed_dicts:
        fields.extend(typed_dict_to_spark_fields(typed_dict))

    return StructType(fields)


# COMMAND ----------

# DBTITLE 1,Parse all the documents and write the table
# Tune this parameter to optimize performance.  
# More partitions will improve performance, but may cause out of 
# memory errors if your cluster is too small.
NUM_PARTITIONS = 50

# Load the Unity Catalog Volume files into a Spark DataFrame
raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_config.volume_path,
).repartition(NUM_PARTITIONS)

# Apply the parsing UDF to the Spark DataFrame
parsed_files_df = apply_parsing_fn(
    raw_files_df=raw_files_df,
    # Modify this function to change the parser, extract additional metadata, etc
    parse_file_fn=file_parser,
    # The schema of the resulting Delta Table will follow the schema defined in ParserReturnValue
    parsed_df_schema=typed_dicts_to_spark_schema(ParserReturnValue),
)

# Write to a Delta Table
parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    output_config.parsed_docs_table
)

# Get resulting table
parsed_files_df = spark.table(output_config.parsed_docs_table)
parsed_files_no_errors_df = parsed_files_df.filter(
    parsed_files_df.parser_status == "SUCCESS"
)

# Show successfully parsed documents
print(f"Parsed {parsed_files_df.count()} / {parsed_files_no_errors_df.count()} documents successfully.  Inspect `parsed_files_no_errors_df` or visit {get_table_url(output_config.parsed_docs_table)} to see all parsed documents, including any errors.")
display(parsed_files_no_errors_df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC Show any parsing failures or successfully parsed files that resulted in an empty document.

# COMMAND ----------

# DBTITLE 1,Define helpers to check the parsed documents
from pyspark.sql import DataFrame

def check_parsed_df_for_errors(parsed_files_df) -> tuple[bool, str, DataFrame]:
    # Check and warn on any errors
    errors_df = parsed_files_df.filter(func.col(f"parser_status") != "SUCCESS")

    num_errors = errors_df.count()
    if num_errors > 0:
        msg = f"{num_errors} documents ({round(errors_df.count()/parsed_files_df.count(), 2)*100}) of documents had parse errors. Please review."
        return (True, msg, errors_df)
    else:
        msg = "All documents were parsed."
        print(msg)
        return (False, msg, None)


def check_parsed_df_for_empty_parsed_files(parsed_files_df):
    # Check and warn on any errors
    num_empty_df = parsed_files_df.filter(
        func.col(f"parser_status") == "SUCCESS"
    ).filter(func.col("content") == "")

    num_errors = num_empty_df.count()
    if num_errors > 0:
        msg = f"{num_errors} documents ({round(num_empty_df.count()/parsed_files_df.count(), 2)*100}) of documents returned empty parsing results. Please review."
        return (True, msg, num_empty_df)
    else:
        msg = "All documents produced non-null parsing results."
        print(msg)
        return (False, msg, None)


# COMMAND ----------

# DBTITLE 1,Check the results of the parsed output

# Any documents that failed to parse
is_error, msg, failed_docs_df = check_parsed_df_for_errors(parsed_files_df)
if is_error:
    display(failed_docs_df.toPandas())
    raise Exception(msg)
    
# Any documents that returned empty parsing results
is_error, msg, empty_docs_df = check_parsed_df_for_empty_parsed_files(parsed_files_df)
if is_error:
    display(empty_docs_df.toPandas())
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ✏️ Step 2: Compute chunks of documents
# MAGIC
# MAGIC In this step, we will split our documents into smaller chunks to index them in our vector database.
# MAGIC
# MAGIC We provide a default implementation of a recursive text splitter.  To create your own chunking logic, adapt the `get_recursive_character_text_splitter()` function defined in one of the prior cells which is called in the following cell.

# COMMAND ----------

# DBTITLE 1,Create the chunking function
# Get the chunking function
recursive_character_text_splitter_fn = get_keyword_based_text_splitter(
    model_serving_endpoint=chunking_config.embedding_model_endpoint,
    keyword="---END---"
)

# Determine which columns to propagate from the docs table to the chunks table.

# Get the columns from the parser except for the content
# You can modify this to adjust which fields are propagated from the docs table to the chunks table.
propagate_columns = [
    field.name
    for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
    if field.name != "content"
]

# If you want to implement retrieval strategies such as presenting the entire document vs. the chunk to the LLM, include `contentich contains the doc's full parsed text.  By default this is not included because the size of contcontentquite large and cause performance issues.
# propagate_columns = [
#     field.name
#     for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
# ]

# COMMAND ----------

# DBTITLE 1,Define a helper to apply chunking
from typing import Literal, Optional, Any, Callable
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from typing import Callable
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from pyspark.sql import DataFrame, SparkSession


def apply_chunking_fn(
    parsed_docs_df: DataFrame,
    chunking_fn: Callable[[str], list[str]],
    propagate_columns: list[str],
    doc_column: str = "content",
) -> DataFrame:
    # imports here to avoid requiring these libraries in all notebooks since the data pipeline config imports this package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    import tiktoken

    print(
        f"Applying chunking UDF to {parsed_docs_df.count()} documents using Spark - this may take a long time if you have many documents..."
    )

    parser_udf = func.udf(
        chunking_fn, returnType=ArrayType(StringType()), useArrow=True
    )
    chunked_array_docs = parsed_docs_df.withColumn(
        "content_chunked", parser_udf(doc_column)
    )  # .drop(doc_column)
    chunked_docs = chunked_array_docs.select(
        *propagate_columns, explode("content_chunked").alias("content_chunked")
    )

    # Add a primary key: "chunk_id".
    chunks_with_ids = chunked_docs.withColumn(
        "chunk_id", func.md5(func.col("content_chunked"))
    )
    # Reorder for better display.
    chunks_with_ids = chunks_with_ids.select(
        "chunk_id", "content_chunked", *propagate_columns
    )

    return chunks_with_ids


# COMMAND ----------

# MAGIC %md
# MAGIC 🚫✏️ Run the chunking function within Spark

# COMMAND ----------

# DBTITLE 1,Chunk all the parsed documents
# Set the TRANSFORMERS_CACHE environment variable to a writable directory
os.environ['TRANSFORMERS_CACHE'] = '/dbfs/tmp/transformers_cache'

# Tune this parameter to optimize performance.  More partitions will improve performance, but may cause out of memory errors if your cluster is too small.
NUM_PARTITIONS = 50

# Load parsed docs
parsed_files_df = spark.table(output_config.parsed_docs_table).repartition(NUM_PARTITIONS)

chunked_docs_df = chunked_docs_table = apply_chunking_fn(
    # The source documents table.
    parsed_docs_df=parsed_files_df,
    # The chunking function that takes a string (document) and returns a list of strings (chunks).
    chunking_fn=recursive_character_text_splitter_fn,
    # Choose which columns to propagate from the docs table to chunks table. `doc_uri` column is required we can propagate the original document URL to the Agent's web app.
    propagate_columns=propagate_columns,
)

# Write to Delta Table
chunked_docs_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(output_config.chunked_docs_table)

# Get resulting table
chunked_docs_df = spark.table(output_config.chunked_docs_table)

# Show number of chunks created
print(f"Created {chunked_docs_df.count()} chunks.  Inspect `chunked_docs_df` or visit {get_table_url(output_config.chunked_docs_table)} to see the results.")

# Enable CDC feed for VS index sync
cdc_results = spark.sql(f"ALTER TABLE {output_config.chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# Show chunks
display(chunked_docs_df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫 ✏️ Step 3: Create the vector index
# MAGIC
# MAGIC In this step, we'll embed the documents to compute the vector index over the chunks and create our retriever index that will be used to query relevant documents to the user question.  The embedding pipeline is handled within Databricks Vector Search using [Delta Sync](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

# COMMAND ----------

# DBTITLE 1,Define a helper to build the index
from databricks.sdk.service.vectorsearch import (
    VectorSearchIndexesAPI,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist, BadRequest
import time

# `build_retriever_index` will build the vector search index which is used by our RAG to retrieve relevant documents.

# Arguments:
# - `chunked_docs_table`: The chunked documents table. There is expected to be a `chunked_text` column, a `chunk_id` column, and a `url` column.
# -  `primary_key`: The column to use for the vector index primary key.
# - `embedding_source_column`: The column to compute embeddings for in the vector index.
# - `vector_search_endpoint`: An optional vector search endpoint name. It not defined, defaults to the `{table_id}_vector_search`.
# - `vector_search_index_name`: An optional index name. If not defined, defaults to `{chunked_docs_table}_index`.
# - `embedding_endpoint_name`: An embedding endpoint name.
# - `force_delete_vector_search_endpoint`: Setting this to true will rebuild the vector search endpoint.


def build_retriever_index(
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vector_search_index_name: str,
    embedding_endpoint_name: str,
    force_delete_index_before_create=False,
    primary_key: str = "chunk_id",  # hard coded in the apply_chunking_fn
    embedding_source_column: str = "content_chunked",  # hard coded in the apply_chunking_fn
) -> tuple[bool, str]:
    # Initialize workspace client and vector search API
    w = WorkspaceClient()
    vsc = w.vector_search_indexes

    def find_index(index_name):
        try:
            return vsc.get_index(index_name=index_name)
        except ResourceDoesNotExist:
            return None

    def wait_for_index_to_be_ready(index):
        while not index.status.ready:
            print(
                f"Index {vector_search_index_name} exists, but is not ready, waiting 30 seconds..."
            )
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    def wait_for_index_to_be_deleted(index):
        while index:
            print(
                f"Waiting for index {vector_search_index_name} to be deleted, waiting 30 seconds..."
            )
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    existing_index = find_index(index_name=vector_search_index_name)
    if existing_index:
        print(f"Found existing index {get_table_url(vector_search_index_name)}...")
        if force_delete_index_before_create:
            print(f"Deleting index {vector_search_index_name}...")
            vsc.delete_index(index_name=vector_search_index_name)
            wait_for_index_to_be_deleted(existing_index)
            create_index = True
        else:
            wait_for_index_to_be_ready(existing_index)
            create_index = False
            print(
                f"Starting the sync of index {vector_search_index_name}, this can take 15 minutes or much longer if you have a larger number of documents."
            )
            # print(existing_index)
            try:
                vsc.sync_index(index_name=vector_search_index_name)
                msg = f"Kicked off index sync for {vector_search_index_name}."
                return (False, msg)
            except BadRequest as e:
                msg = f"Index sync already in progress, so failed to kick off index sync for {vector_search_index_name}.  Please wait for the index to finish syncing and try again."
                return (True, msg)
    else:
        print(
            f'Creating new vector search index "{vector_search_index_name}" on endpoint "{vector_search_endpoint}"'
        )
        create_index = True

    if create_index:
        print(
            "Computing document embeddings and Vector Search Index. This can take 15 minutes or much longer if you have a larger number of documents."
        )
        try:
            # Create delta sync index spec using the proper class
            delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
                source_table=chunked_docs_table_name,
                pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name=embedding_source_column,
                        embedding_model_endpoint_name=embedding_endpoint_name,
                    )
                ],
            )

            vsc.create_index(
                name=vector_search_index_name,
                endpoint_name=vector_search_endpoint,
                primary_key=primary_key,
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=delta_sync_spec,
            )
            msg = (
                f"Successfully created vector search index {vector_search_index_name}."
            )
            print(msg)
            return (False, msg)
        except Exception as e:
            msg = f"Vector search index creation failed. Wait 5 minutes and try running this cell again."
            return (True, msg)


# COMMAND ----------

# DBTITLE 1,Build the index
is_error, msg = retriever_index_result = build_retriever_index(
    # Spark requires `` to escape names with special chars, VS client does not.
    chunked_docs_table_name=output_config.chunked_docs_table.replace("`", ""),
    vector_search_endpoint=output_config.vector_search_endpoint,
    vector_search_index_name=output_config.vector_index,

    # Must match the embedding endpoint you used to chunk your documents
    embedding_endpoint_name=chunking_config.embedding_model_endpoint,

    # Set to true to re-create the vector search endpoint when re-running the data pipeline.  If set to True, syncing will not work if re-run the pipeline and change the schema of chunked_docs_table_name.  Keeping this as False will allow Vector Search to avoid recomputing embeddings for any row with that has a chunk_id that was previously computed.
    force_delete_index_before_create=False,
)
if is_error:
    raise Exception(msg)
else:
    print("NOTE: This cell will complete before the vector index has finished syncing/embedding your chunks & is ready for queries!")
    print(f"View sync status here: {get_table_url(output_config.vector_index)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫 ✏️ Print links to view the resulting tables/index

# COMMAND ----------

# DBTITLE 1,Print links to the output artifacts in Unity Catalog
print(f"Parsed docs table: {get_table_url(output_config.parsed_docs_table)}\n")
print(f"Chunked docs table: {get_table_url(output_config.chunked_docs_table)}\n")
print(f"Vector search index: {get_table_url(output_config.vector_index)}\n")