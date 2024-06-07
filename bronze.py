# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col
from pyspark.sql.functions import split, size, col
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import when
from pyspark.sql.functions import countDistinct
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------BRONZE LEVEL------------------------------------

# Create a Spark session, connect to HDFS cluster
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

# Load Titles dataset from HDFS
titles_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", "\t") \
    .load("title_basics.tsv") 
titles_df

# Load ratings dataset from HDFS
ratings_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", "\t") \
    .load("title_ratings.tsv")
ratings_df

# Print the schema of the both df
titles_df.printSchema()
ratings_df.printSchema()

# Describe both df
titles_df.describe().show()
ratings_df.describe().show()

# Count distinct values in ID Ratings vs total row number
dist_tconst_r = ratings_df.select(countDistinct("tconst")).collect()[0][0]
print("Exact distinct count:", dist_tconst_r)
total_r = ratings_df.count()
print("Total count:", total_r)

# Count distinct values in ID Titles vs total row number
dist_tconst_t = titles_df.select(countDistinct("tconst")).collect()[0][0]
print("Exact distinct count:", dist_tconst_t)
total_t = titles_df.count()
print("Total count:", total_t)

# Check number of NULL values in Titles
null_counts = []
for column in titles_df.columns:
    null_counts.append(sum(col(column).isNull().cast("int")).alias(column))
null_counts_df = titles_df.select(null_counts)
null_counts_df.show()

# Check number of NULL values in Ratings
null_counts = []
for column in ratings_df.columns:
    null_counts.append(sum(col(column).isNull().cast("int")).alias(column))
null_counts_df = ratings_df.select(null_counts)
null_counts_df.show()

# Replace "\\N" values by NULL and transform columns into INT format
columns_to_process = ["startYear", "endYear", "runtimeMinutes"]
for column in columns_to_process:
    titles_df = titles_df.withColumn(column, when(titles_df[column] == "\\N", None).otherwise(titles_df[column].cast("int")))
# Replace "\\N" values by NULL in string format columns
columns_to_process = ["tconst", "titleType", "primaryTitle", "originalTitle", "genres"]
for column in columns_to_process:
    titles_df = titles_df.withColumn(column, when(titles_df[column] == "\\N", None).otherwise(titles_df[column].cast("string")))
titles_df

# Re-chech the amount of null values in df after the transformation
null_counts = []
for column in titles_df.columns:
    null_counts.append(sum(col(column).isNull().cast("int")).alias(column))
null_counts_df = titles_df.select(null_counts)
null_counts_df.show()

# Check for duplicates in Titles
count_t = titles_df.count()
deduplicated_df_t = titles_df.dropDuplicates()
deduplicated_count_t = deduplicated_df_t.count()
if count_t == deduplicated_count_t:
    print("No duplicates found.")
else:
    print("Duplicates found and removed.")
deduplicated_df_t.show()

# Check for duplicates in Ratings
count_r = ratings_df.count()
deduplicated_df_r = ratings_df.dropDuplicates()
deduplicated_count_r = deduplicated_df_r.count()
if count_r == deduplicated_count_r:
    print("No duplicates found.")
else:
    print("Duplicates found and removed.")
deduplicated_df_r.show()

# Split Genre column in titles_df on 3 columns
titles_df = titles_df.withColumn("genres_array", split(col("genres"), ","))
max_values = titles_df.select(size(col("genres_array")).alias("num_values")).agg({"num_values": "max"}).collect()[0][0]
print(f"Maximal number of values in a row: {max_values}")
for i in range(max_values):
    titles_df = titles_df.withColumn(f"genre_{i+1}", col("genres_array").getItem(i))
titles_df = titles_df.drop("genres_array")
titles_df.show(truncate=False)
titles_df = titles_df.drop("genres")
titles_df


# Export titles_df into HDFS /bronze/
# Export ratings_df into HDFS /bronze/
