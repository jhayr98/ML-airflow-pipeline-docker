import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml import MLSystem

class TestMLSystem(unittest.TestCase):

    def setUp(self):
        # crear un df para usarlo de dato dummy
        self.data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'person_age': [25, 30, 40, 50],
            'person_emp_length': [5, 10, 15, 20],
            'loan_amnt': [10000, 20000, 30000, 40000],
            'loan_int_rate': [10, 15, 20, 25],
            'loan_status': [1, 0, 1, 0]
        })
        
        self.new_data = pd.DataFrame({
            'id': [5, 6],
            'person_age': [27, 35],
            'person_emp_length': [6, 11],
            'loan_amnt': [12000, 18000],
            'loan_int_rate': [9, 12]
        })

        # instanciar sistema ML
        self.ml_system = MLSystem()

    def test_preprocess_data(self):
        # probamos el metodo  preprocess_data
        features, target, df_id = self.ml_system.preprocess_data(self.data)
        
        # verificamos que las características y el target no sean nulas
        self.assertIsNotNone(features)
        self.assertIsNotNone(target)
        
        # verificamos  que las dimensiones sean correctas
        self.assertEqual(features.shape[0], self.data.shape[0])
        self.assertEqual(df_id.shape[0], self.data.shape[0])
        
        # verificamos que los datos se escalan correctamente
        self.assertTrue((features.select_dtypes(include=[np.float64]).values <= 1).all())
        self.assertTrue((features.select_dtypes(include=[np.float64]).values >= -1).all())
    
    def test_train(self):
        # probar el método train
        model = self.ml_system.train(self.data)
        
        # comprobar  que el modelo no sea nulo después de entrenar
        self.assertIsNotNone(model)
        self.assertIsInstance(model, RandomForestClassifier)
    
    def test_evaluate(self):
        # probar del método evaluate , no deberia arrojar errores
        self.ml_system.train(self.data)
        try:
            self.ml_system.evaluate()
            evaluation_successful = True
        except Exception as e:
            evaluation_successful = False
            print("Error during evaluation:", e)
        
        self.assertTrue(evaluation_successful)
    
    def test_predict(self):
        # Verificación del método predict
        self.ml_system.train(self.data)
        predictions_df = self.ml_system.predict(self.new_data)
        
        # verificar que el DataFrame de resultados tenga la columna 'prediction'
        self.assertIn('prediction', predictions_df.columns)
        
        # revisar  que el número de filas coincida con el de los datos nuevos
        self.assertEqual(predictions_df.shape[0], self.new_data.shape[0])

# ejecutar
if __name__ == '__main__':
    unittest.main()
