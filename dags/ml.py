import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score


class MLSystem:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=777, n_estimators=450 , max_depth= 40)
        self.features_train = None
        self.target_train = None
        self.df_id_train= None
        self.features_test = None
        self.target_test = None
        self.df_id_test= None
    
    def preprocess_data(self,data):
        data=data[(data['person_age'] >= 20) & (data['person_age'] <= 60)]
        data=data[(data['person_emp_length'] >= 0) & (data['person_emp_length'] <= 30)]
        data=data[(data['loan_amnt'] >= 0) & (data['loan_amnt'] <= 40000)]
        data=data[(data['loan_int_rate'] >= 5) & (data['loan_int_rate'] <= 25)]
        df_id =data['id']
        data=data.drop(['id'],axis=1)
        # ONE HOT ENCODING
        data_ohe = pd.get_dummies(data, drop_first=True, dtype='int')
        # separar los el dataset
        if 'loan_status' in data_ohe.columns:
            features = data_ohe.drop(['loan_status'], axis=1)
            target = data_ohe['loan_status']
            # escalado
            numeric = data.drop(['loan_status'],axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
        else:
            features = data_ohe.copy()
            target = pd.Series(dtype='int')
            numeric = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        global scaler
        scaler= StandardScaler()
        scaler.fit(features[numeric])
        features[numeric] = scaler.transform(features[numeric])
        return features, target , df_id

    def train (self,data):
        self.features_train, self.target_train , self.df_id =  self.preprocess_data(data)
        # Entrenar el modelo
        self.model.fit(self.features_train, self.target_train)
        return self.model

    def evaluate(self):
        # Hacer predicciones
        prediction = self.model.predict(self.features_train)
        # Imprimir el informe de clasificación
        print('accuracy: ',accuracy_score(self.target_train, prediction))

    def predict(self, new_data):
        # Preprocesar nuevos datos antes de la predicción
        features, target , df_id = self.preprocess_data(new_data)
        new_prediction = self.model.predict(features)
        
        # Convertir las predicciones en un DataFrame
        predictions_df = pd.DataFrame(new_prediction, columns=['prediction'])

        # Combinar df_id y las predicciones en un solo DataFrame
        result_df = pd.concat([df_id.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

        return result_df
        
