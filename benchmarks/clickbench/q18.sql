SELECT UserID, extract(minute FROM make_timestamp(EventTime * 1000000)) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;
