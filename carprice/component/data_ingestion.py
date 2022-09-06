from carprice.entity.config_entity import DataIngestionConfig
import sys,os
from carprice.exception import CarException
from carprice.logger import logging
from carprice.entity.artifact_entity import DataIngestionArtifact
import numpy as np
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CarException(e,sys)
    

    def download_carprice_data(self) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            download_dir = self.data_ingestion_config.raw_data_dir
            
            os.makedirs(download_dir,exist_ok=True)

            filename = os.path.basename(download_url)

            raw_data_dir = os.path.join(download_dir, filename)

            logging.info(f"Downloading file from :[{download_url}] into :[{raw_data_dir}]")
            urllib.request.urlretrieve(download_url, raw_data_dir)
            logging.info(f"File :[{raw_data_dir}] has been downloaded successfully.")
            return raw_data_dir

        except Exception as e:
            raise CarException(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            data_file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{data_file_path}]")
            data_frame = pd.read_csv(data_file_path, index_col=[0])
            data_frame.drop("car_name", axis=1, inplace=True)
            
            logging.info(f"Splitting data into train and test")
            train_set = None
            test_set = None

            train_set, test_set = train_test_split(data_frame, test_size=0.2, random_state=42)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path,index=False)

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path,index=False)
            
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise CarException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            raw_data_dir =  self.download_carprice_data()
            return self.split_data_as_train_test()
        except Exception as e:
            raise CarException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
