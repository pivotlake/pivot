SELECT CAST(j.commit.collection AS VARCHAR) AS event, count(*) AS count FROM bluesky GROUP BY event ORDER BY count DESC, event;
