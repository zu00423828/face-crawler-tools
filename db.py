import sqlite3

from sqlalchemy import table


class CrawlerDB():
    def __init__(self, db_path):
        self.connect = sqlite3.connect(db_path, isolation_level=None)
        cursor = self.connect.cursor()
        table_schema = '''
            CREATE TABLE IF NOT EXISTS img_table (
            id  INTEGER  PRIMARY KEY AUTOINCREMENT,
            keyword VARCHAR(10),
            img_url TEXT(300) UNIQUE
            )'''
        cursor.execute(table_schema)
        cursor.close()

    def insert_data(self, keyword, img_url):
        sql_statement = f"INSERT INTO img_table (keyword,img_url) VALUES('{keyword}','{img_url}')"
        cursor = self.connect.cursor()
        cursor.execute(sql_statement)
        cursor.close()

    def check_exists(self, img_url):
        sql_statement = f"SELECT id FROM img_table WHERE img_url='{img_url}'"
        cursor = self.connect.cursor()
        cursor.execute(sql_statement)
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return False
        return True
