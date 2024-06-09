
# Create Spark session
spark = SparkSession.builder \
    .appName("Silver_to_Gold") \
    .getOrCreate()

# Load Titles dataset from HDFS /silver/
csv_t = "hdfs://namenode/user/output/imdbdata/silver/titles_df.csv"
titles_df = spark.read.csv(csv_t, header=True, inferSchema=True)
titles_df.show(5)

# Load Ratings dataset from HDFS /silver/
csv_r = "hdfs://namenode/user/output/imdbdata/silver/ratings_df.csv"
ratings_df = spark.read.csv(csv_r, header=True, inferSchema=True)
ratings_df.show(5)

# Merge both tables (we want to keep all the rows from ratings_df table because it's has rating by each episode of a TV-show and not only by title)
df = titles_df.join(ratings_df, on="tconst", how="right")
df

#Check number of rows after the merge
print(df.count())
print(ratings_df.count())


# Some visual data exploraition with plots

# Average Ratings histogram
average_ratings = ratings_df.select('averageRating').toPandas()['averageRating']
plt.figure(figsize=(8, 6))
sns.histplot(average_ratings, bins=20, kde=True, color='skyblue')
plt.title('Histogram of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# Save the plot locally
local_plot1 = "/tmp/Average Ratings histogram.png"
plt.savefig(local_plot1)
plt.close()


# Genre count barplot
genre_counts = titles_df.groupBy('genre_1').count()
genre_counts = genre_counts.orderBy('count', ascending=False)
genre_counts_list = genre_counts.collect()
genres = [row['genre_1'] for row in genre_counts_list]
counts = [row['count'] for row in genre_counts_list]
plt.figure(figsize=(10, 6))
sns.barplot(x=counts, y=genres, palette='viridis')
plt.title('Number of Titles by Genre')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()

# Save the plot locally
local_plot2 = "/tmp/Genre count barplot.png"
plt.savefig(local_plot2)
plt.close()


# Type count barplot
type_counts = titles_df.groupBy('titleType').count()
type_counts = type_counts.orderBy('count', ascending=False)
type_counts_list = type_counts.collect()
genres = [row['titleType'] for row in type_counts_list]
counts = [row['count'] for row in type_counts_list]
plt.figure(figsize=(10, 6))
sns.barplot(x=counts, y=genres, palette='viridis')
plt.title('Number of Titles by Type')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()

# Save the plot locally
local_plot2 = "/tmp/Type count barplot.png"
plt.savefig(local_plot2)
plt.close()


# Correlation Heatmap
pandas_df = df.select('averageRating', 'numVotes', 'startYear', 'endYear', 'runtimeMinutes', 'isAdult').toPandas()
plt.figure(figsize=(10, 8))
correlation_matrix = pandas_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Save the plot locally
local_plot3 = "/tmp/Correlation Heatmap.png"
plt.savefig(local_plot3)
plt.close()


# Export df into HDFS /gold/
df.write.csv("hdfs://your_hdfs_cluster/path/to/imdbdata/gold", header=True, mode="overwrite")
