import pandas as pd
import pickle
import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load

import pickle

from ml import MLSystem

# parametros, hay mas
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 12),
    'email ':['jhayr1998@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# definicion del dag, 
dag = DAG(
    'ml_workflow_jhayr_v2',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)

def load_data(ti):
    data_path = '/opt/airflow/dags/data/train.csv'
    data_path_test= '/opt/airflow/dags/data/test.csv'
    
    df = pd.read_csv(data_path)
    df_test=pd.read_csv(data_path_test)

    # serializar el df a JSON como string
    ti.xcom_push(key='train_data', value=df.to_json(orient='records'))
    ti.xcom_push(key='test_data', value=df_test.to_json(orient='records'))



def auto_ml (ti):
    json_data = ti.xcom_pull(task_ids='load_data', key='train_data')
    df = pd.read_json(json_data)
    json_data = ti.xcom_pull(task_ids='load_data', key='test_data')
    df_test = pd.read_json(json_data)

    ml = MLSystem()
    # entrenar model
    model=ml.train(df)
    # evaluarg
    ml.evaluate()
    # hacer predccion con el test dataset
    df_sample_submission=ml.predict(df_test)

    model_path= '/opt/airflow/dags/data/model_resutado.joblib'
    dump(model,model_path)

    ti.xcom_push(key='sample_submission', value=df_sample_submission.to_json(orient='records'))

def save_load_model(ti):
    json_data = ti.xcom_pull(task_ids='auto_ml', key='sample_submission')
    df_sample_submission = pd.read_json(json_data)

    save_path = '/opt/airflow/dags/data/sample_submission_resultado.csv'
    df_sample_submission.to_csv(save_path, index=False)


    
# definir las tareas, son las llamadas a las funciones con los operadores

load_data_task= PythonOperator(
    task_id= 'load_data',
    python_callable= load_data,
    dag=dag
)

auto_ml_task = PythonOperator(
    task_id='auto_ml',
    python_callable=auto_ml,
    dag=dag,
)

save_load_model_task = PythonOperator(
    task_id='save_load_model',
    python_callable=save_load_model,
    dag=dag,
)


# orquestacion
load_data_task >> auto_ml_task >> save_load_model_task 

