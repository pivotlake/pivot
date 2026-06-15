SELECT UserID, SearchPhrase, COUNT(*) AS c FROM hits GROUP BY UserID, SearchPhrase ORDER BY c DESC, UserID, SearchPhrase LIMIT 10;
