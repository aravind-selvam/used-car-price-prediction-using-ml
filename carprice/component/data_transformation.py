import sys
import os
import numpy as np
import pandas as pd
from cgi import test
from sklearn import preprocessing
from category_encoders.binary import BinaryEncoder
from carprice.exception import CarException
from carprice.logger import logging
from carprice.entity.config_entity import DataTransformationConfig
from carprice.entity.artifact_entity import DataIngestionArtifact,\
    DataValidationArtifact, DataTransformationArtifact
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from carprice.constant import *
from carprice.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(
                f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CarException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            onehot_columns = dataset_schema[ONEHOT_COLUMNS_KEY]
            binary_columns = dataset_schema[BINARY_COLUMNS_KEY]
            
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            binary_transformer = BinaryEncoder()

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, onehot_columns),
                    ("BinaryEncoder", binary_transformer, binary_columns),
                    ("StandardScaler", numeric_transformer, numerical_columns)
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            return preprocessor

        except Exception as e:
            raise CarException(e, sys) from e
    
    def _outlier_capping(self, col, df):
        try: 
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col]= upper_limit
            df.loc[(df[col]<lower_limit), col]= lower_limit 
            return df
        
        except Exception as e:
            raise CarException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(
                f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path,
                                 schema_file_path=schema_file_path)

            test_df = load_data(file_path=test_file_path,
                                schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]
            numerical_columns = schema[NUMERICAL_COLUMN_KEY]
            
            continuous_columns=[feature for feature in numerical_columns if len(train_df[feature].unique())>=25]

            for col in continuous_columns:
                self._outlier_capping(col=col, df= train_df)
            
            logging.info(f"Outlier capped in train df")
            
            for col in continuous_columns:
                self._outlier_capping(col=col, df= test_df)
            logging.info(f"Outlier capped in test df ")

            
            logging.info(
                f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(
                train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(
                test_file_path).replace(".csv", ".npz")

            transformed_train_file_path = os.path.join(
                transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(
                transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")

            save_numpy_array_data(
                file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(
                file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,
                        obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfull.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path
                                                                      )
            logging.info(
                f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CarException(e, sys) from e
    
    def __del__(self):
        logging.info(
            f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
