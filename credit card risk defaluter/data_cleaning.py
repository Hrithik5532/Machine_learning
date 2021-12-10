
from typing import ValuesView
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class Cleaning:
    

    def remove_columns(  data,columns):
        data=data
        columns=columns
        try:
            useful_data= data.drop(labels= columns, axis=1) # drop the labels specified in the columns
        
            return useful_data
        except Exception as e:
           
            raise Exception()
    
    def separate_label_feature(   data, label_column_name):
    
        try:
            X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            Y=data[label_column_name] # Filter the Label columns
            return  X, Y
        except Exception as e:
            raise Exception()

    def dropUnnecessaryColumns(data,columnNameList):
        #drop unwanted columns
        data = data.drop(columnNameList,axis=1)
        return data

    def replaceInvalidValuesWithNull(  data):
        # replace unnecessary values to na
        for column in data.columns:
            count = data[column][(data[column] == 'XNA')].count()  
            if count != 0:
                data[column] = data[column].replace('XNA', np.nan)
        return data

    
    def is_null_present( data):
      
         null_present = False
         cols_with_missing_values=[]
         cols = data.columns
         try:
            null_counts= data.isna().sum() # check for the count of null values per column
            for i in range(len( null_counts)):
                if  null_counts[i]>0:
                     null_present=True
                     cols_with_missing_values.append( cols[i])
            
            return  null_present,  cols_with_missing_values
         except Exception as e:
            raise Exception()


    def encodeCategoricalValues(  data):
    
        data["class"] = data["class"].map({'p': 1, 'e': 2})

        for column in data.drop(['class'],axis=1).columns:
            data = pd.get_dummies(data, columns=[column])

        return data

    def encodeCategoricalValuesPrediction(data,col_list):
        for column in (col_list):
            col=pd.get_dummies(data[column],drop_first=True)
            data = data.drop(columns=column, axis=1)
            data = pd.concat([data,col], axis=1)
        return data


    def standardScalingData(X,y):
    
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)
        y_scaled = scalar.fit(y)
        return X_scaled, y_scaled

    def fill_unknown_values(data):

        columns = []
        for col in data.columns:
            if data[col].dtype == "object":
                columns.append(col)
    
        for col in columns:
            data[col] = data[col].fillna(method ="ffill")
    
        no_obj_data = data.drop(columns=columns, inplace=False).copy()

        for col in no_obj_data.columns:
            data[col] = data[col].fillna(value = data[col].mean())
        return data