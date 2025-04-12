# Databricks notebook source
# MAGIC %md
# MAGIC # Libraries

# COMMAND ----------

from pyspark.sql.functions import when, col, regexp_replace, to_date
from pyspark.sql.types import IntegerType, FloatType, DateType

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the Files from Volume

# COMMAND ----------

path = '/Volumes/dts_proves_pre/startups_list/startups_catalog/startups_result.csv'

# COMMAND ----------

df= spark.read.options(delimiter=";", header=True, encoding='utf-8').csv(path)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualització de les dades

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dades Mestres

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the Spark DataFrame as a table in the catalog with Column Mapping mode enabled

# COMMAND ----------

# MAGIC %md
# MAGIC By default, the columns of a table cannot have some special characters, like the space " ". With Column Mapping mode enabled, we can allow these special characters.

# COMMAND ----------

output_path= 'dts_proves_pre.`startups_list`.`startups_catalog`'


# COMMAND ----------

df.write.option("delta.columnMapping.mode", "name").saveAsTable(output_path)