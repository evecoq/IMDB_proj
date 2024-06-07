#--------------------------------SILVER LEVEL-----------------------------

# Load Titles dataset from HDFS /bronze/
# Load Ratings dataset from HDFS /bronze/


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

# Correlation Heatmap
pandas_df = df.select('averageRating', 'numVotes', 'startYear', 'endYear', 'runtimeMinutes', 'isAdult').toPandas()
plt.figure(figsize=(10, 8))
correlation_matrix = pandas_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Export df into HDFS /silver/
