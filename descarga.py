import os
os.environ['KAGGLE_USERNAME'] = 'jhayrsanchezgarcia'
os.environ['KAGGLE_KEY'] = 'c1ee0f1a522ee30e78203d1288a53f23'

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
api = KaggleApi()
api.authenticate()

# Download the competition files
competition_name = 'playground-series-s4e4'
download_path = 'data/'
api.competition_download_files(competition_name, path=download_path)

# Unzip the downloaded files
for item in os.listdir(download_path):
    if item.endswith('.zip'):
        zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
        zip_ref.extractall(download_path)
        zip_ref.close()
        print(f"Unzipped {item}")