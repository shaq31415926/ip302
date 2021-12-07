import pandas as pd


def identify_missing_data(df):
    """
    This function is used to identify missing data
    
    @param df pandas DataFrame
    
    @return a DataFrame with the percentage of missing data for every feature and the data types
    """
    
    percent_missing = df.isnull().mean()
    
    missing_value_df = pd.DataFrame(percent_missing).reset_index() # convert to DataFrame
    missing_value_df = missing_value_df.rename(columns = {"index" : "feature",
                                                                0 : "percent_missing"}) # rename columns

    missing_value_df = missing_value_df.sort_values(by = ['percent_missing'], ascending = False) # sort the values
    
    data_types_df = pd.DataFrame(df.dtypes).reset_index().rename(columns = {"index" : "feature",
                                                                0 : "data_type"}) # rename columns
    
    missing_value_df = missing_value_df.merge(data_types_df, on = "feature") # join the dataframe with datatype
    
    missing_value_df.percent_missing = round(missing_value_df.percent_missing*100, 2) # format the percent_missing
    
    return missing_value_df


def handle_categorical_missing_data(df):
    """
    This definition can be used to replace all categorical data with Unknown
    
    @param df pandas DataFrame
    
    @returns nothing - replaces all the categorical missing data with Unknown
    """
    
    missing_value_df = identify_missing_data(df)
    missing_value_df_reduced = missing_value_df[missing_value_df.percent_missing > 0]
    categorical_cols = list(missing_value_df_reduced[missing_value_df_reduced.data_type == 'object'].feature)
    print("number of categorical cols with missing data:", len(categorical_cols))
    
    for c in categorical_cols:
        print("replacing missing values for:", c)
        df[c].fillna('Unknown', inplace=True)
        
        
def handle_numerical_missing_data(df, fill_na_value):
    """
    This definition can be used to replace all categorical data with Unknown
    
    @param df pandas DataFrame
    @param fill_na_value float - the value you want to use to replace nas with

    
    @returns nothing - replaces all the numerical missing data with a value of your choice
    """
    
    missing_value_df = identify_missing_data(df)
    missing_value_df_reduced = missing_value_df[missing_value_df.percent_missing > 0]
    numerical_cols = list(missing_value_df_reduced[(missing_value_df_reduced.data_type == 'float64') |(missing_value_df_reduced.data_type == 'int64')].feature)
    print("number of numerical cols with missing data:", len(numerical_cols))
    
    for n in numerical_cols:
                                                   
        print("replacing missing values for:", n)
        df[n].fillna(fill_na_value, inplace=True)