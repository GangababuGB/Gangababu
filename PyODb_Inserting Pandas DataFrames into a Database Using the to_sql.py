# # Inserting Pandas DataFrames into a Database Using the to_sql
import cx_Oracle
import pandas as pd
from sqlalchemy import create_engine

cx_Oracle.init_oracle_client(lib_dir=r"C:\Oracle\instantclient_21_3")
engine = create_engine('oracle+cx_oracle://TRAINEE19:TRAINEE19@192.168.0.99:1521/?service_name=xyzclient')

# # Prepare Data
irisData = pd.read_csv('https://github.com/Muhd-Shahid/Write-Raw-File-into-Database-Server/raw/main/iris.csv', index_col=False)
# Insert whole DataFrame into Oracle DB
irisData.to_sql('iris2', engine, if_exists='replace', index=False)
print("Record inserted successfully")

# # Query the database to check our work
conn = engine.connect()
data = conn.execute("SELECT * FROM iris2")
iris_df = pd.DataFrame(data.fetchall())
iris_df.columns = data.keys()
print(iris_df.head())
conn.close()
