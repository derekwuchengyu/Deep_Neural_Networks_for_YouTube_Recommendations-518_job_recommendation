import os,sys
import numpy as np
import pandas as pd
from clickhouse_driver import Client
from clickhouse_driver import connect
client = Client('192.168.1.42')
conn = connect('clickhouse://192.168.1.42')
cursor = conn.cursor()

import time 


if __name__ == "__main__":
    print("開始匯入")
    db_names = ['job_basic','profile_basic','profile_condition','profile_description',
                'profile_experience','resume_submit_active','profile_match']

    total_start = time.time()
    for name in db_names:
        start = time.time()
        db_name = f"518paper_{name}"
               
        sql = f"DROP TABLE IF EXISTS  recomm.{db_name}"
        cursor.execute(sql)
        cursor.fetchall()
        sql =f"CREATE TABLE recomm.{db_name} ENGINE = ReplacingMergeTree  \
        ORDER BY id AS  \
        SELECT * FROM mysql('192.168.1.31', 't_518_bi', '{name}', 'bi', '!qaz2wsx')"

        client.execute(sql)
        end = time.time()

        print("匯入:" ,db_name , ", 執行時間:" ,(end - start))
    print("總執行時間: (%.2f seconds)" %(time.time() - total_start))