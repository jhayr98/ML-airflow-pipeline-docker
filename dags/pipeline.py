import pandas as pd

import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load




# parametros, hay mas
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 13),
    'email ':['jhayr1998@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# definicion del dag, 
dag = DAG(
    'ml_workflow_jhayr',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)






def load_data():
    data_path= '/opt/airflow/dags/data/train.csv'
    df = pd.read_csv(data_path)
    return df

def preprocess_data(ti):
    df = ti.xcom_pull(task_ids='load_data')

    df=df[(df['person_age'] >= 20) & (df['person_age'] <= 60)]
    df=df[(df['person_income'] >= 0) & (df['person_income'] <= 300000)]
    df=df[(df['person_emp_length'] >= 0) & (df['person_emp_length'] <= 30)]
    df=df[(df['loan_amnt'] >= 0) & (df['loan_amnt'] <= 40000)]
    df=df[(df['loan_int_rate'] >= 5) & (df['loan_int_rate'] <= 25)]
    df_id =df['id']
    df=df.drop(['id'],axis=1)
    # ONE HOT ENCODING
    data_ohe = pd.get_dummies(df, drop_first=True, dtype='int')
    # separar los el dataset
    features =data_ohe.drop(['loan_status'],axis=1)
    target = data_ohe['loan_status']
    # escalado
    numeric = df.drop(['loan_status'],axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    global scaler
    scaler= StandardScaler()
    scaler.fit(features[numeric])
    features[numeric] = scaler.transform(features[numeric])
    return features,target,df_id

def train_model(ti):
    features,target,df_id=ti.xcom_pull(task_ids='preprocess_data')
    model= RandomForestClassifier(random_state=777, n_estimators=450 , max_depth= 40)
    model.fit(features,target)
    model_path= '/opt/airflow/dags/data/model.joblib'
    dump(model,model_path)
    return model_path

def evaluate_model(ti):
    model_path= ti.xcom_pull(task_ids='train_model')
    model= load(model_path)
    features,target,df_id=ti.xcom_pull(task_ids='preprocess_data')
    predictions = model.predict(features)
    accuracy=  accuracy_score(target, predictions)
    return accuracy

# definir las tareas, son las llamadas a las funciones con los operadores

load_data_task= PythonOperator(
    task_id= 'load_data',
    python_callable= load_data,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# orquestacion
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task

