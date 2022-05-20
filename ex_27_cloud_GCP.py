from google.cloud import bigquery
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    client = bigquery.Client.from_service_account_json("C:/Users/Anna/.ssh/genial-upgrade-172805-e0e634c95ae9.json")
    query = """SELECT corpus AS title, COUNT(word) AS unique_words
        FROM `bigquery-public-data.samples.shakespeare`
        GROUP BY title
        ORDER BY unique_words
        DESC LIMIT 10"""
    results = client.query(query)

    for row in results:
        title = row['title']
        unique_words = row['unique_words']
        print(f'{title:<20} | {unique_words}')
