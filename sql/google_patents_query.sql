-- Google Patents Public Data, BigQuery sandbox, run 24 November 2026 (15.49 GB processed; 117,835 rows)
SELECT application_number AS app, filing_date AS fd, publication_date AS pd,
  ARRAY_TO_STRING(ARRAY(SELECT DISTINCT SUBSTR(i.code,1,4) FROM UNNEST(ipc) i), ';') AS sub,
  ARRAY_TO_STRING(ARRAY(SELECT DISTINCT CAST(MOD(ABS(FARM_FINGERPRINT(a.name)), 100000000) AS STRING)
                        FROM UNNEST(assignee_harmonized) a), '|') AS asg
FROM `patents-public-data.patents.publications`
WHERE country_code = 'CN' AND kind_code = 'A' AND filing_date BETWEEN 20140101 AND 20251231
  AND EXISTS (SELECT 1 FROM UNNEST(ipc) i WHERE i.code LIKE 'B25J%')
QUALIFY ROW_NUMBER() OVER (PARTITION BY application_number ORDER BY publication_date) = 1
