import pandas as pd
import numpy as np
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import avg


def clean_and_process_temperature_data(temperatures_data_path, save_local=True, fs=None, save_to_s3=False, output_path=None):
    '''
    Cleans and processes the temperature data.
    If the s3 arguments are provided, also saves the final temperature table to s3.
    '''

    temperatures_df = pd.read_csv(temperatures_data_path)

    # format the date and extract the year and month
    temperatures_df['dt'] = pd.to_datetime(temperatures_df['dt'], format='%Y-%m-%d')
    temperatures_df['year'] = temperatures_df['dt'].dt.year
    temperatures_df['month'] = temperatures_df['dt'].dt.month

    temperatures_df.sort_values(['year', 'month'], ascending=[True, True], inplace=True)

    # Remove null values in the AverageTemperature column
    temperatures_df = temperatures_df[pd.notnull(temperatures_df['AverageTemperature'])]

    # Filter on the US
    temperatures_US = temperatures_df[temperatures_df['Country'] == 'United States']

    # Keep the last information
    temperatures_US.drop_duplicates(subset=['City', 'month'], keep='last', inplace=True)

    # Create the temperature id
    temperatures_US.index = np.arange(1, len(temperatures_US) + 1)
    temperatures_US = temperatures_US.reset_index()
    temperatures_US = temperatures_US.rename(columns={'index': 'temperature_id'})

    # Keeping the useful information
    temperatures_US = temperatures_US[['temperature_id', 'City', 'month', 'AverageTemperature']]

    if save_local:
        temperatures_US.to_csv('temperature_table.csv', sep=';', index=False)

    if save_to_s3:
        with fs.open(output_path + 'temperature_table.csv', 'wb') as output_file:
            temperatures_US.to_csv(output_file, sep=';', index=False)

    return temperatures_US


def clean_and_process_us_cities_demographics_data(us_cities_demographics_path, save_local=True, fs=None, save_to_s3=False, output_path=None):
    '''

    Cleans and processes the us demographics data.
    If the s3 arguments are provided, also saves the final demographics table to s3 and/or locally.
    '''

    us_cities_demographics_df = pd.read_csv(us_cities_demographics_path, sep=';')

    # deduplicate the lines
    us_cities_demographics_deduplicated = us_cities_demographics_df.drop_duplicates(subset=['City', 'State'])

    # create city_id for the join with the immigration table
    us_cities_demographics_deduplicated['City'] = us_cities_demographics_deduplicated['City'].str.lower()
    us_cities_demographics_deduplicated['city_id'] = us_cities_demographics_deduplicated[['City', 'State Code']].apply(
        lambda x: hash(tuple(x)), axis=1)

    # keep only the useful columns
    us_cities_demographics_deduplicated_final = us_cities_demographics_deduplicated[
        ['city_id', 'City', 'State', 'State Code', 'Median Age', 'Male Population',
         'Female Population', 'Total Population', 'Foreign-born', 'Average Household Size']]

    if save_local:
        us_cities_demographics_deduplicated_final.to_csv('demographics_table.csv', sep=';', index=False)

    if save_to_s3:
        with fs.open(output_path + 'demographics_table.csv', 'wb') as output_file:
            us_cities_demographics_deduplicated_final.to_csv(output_file, sep=';', index=False)

    return us_cities_demographics_deduplicated


def clean_and_process_immigration_data(immigration_data_path, save_local=True, fs=None, save_to_s3=False, output_path=None):
    '''

    Cleans and processes the us immigration data.
    If the s3 arguments are provided, also saves the final immigration table to s3 and/or locally.
    '''

    spark = SparkSession.builder. \
        config("spark.jars.packages", "saurfang:spark-sas7bdat:2.0.0-s_2.11") \
        .enableHiveSupport().getOrCreate()

    final_df_spark = spark.read.format('com.github.saurfang.sas.spark').load(
        immigration_data_path+'i94_jan16_sub.sas7bdat')
    final_df_spark_columns = list(final_df_spark.columns)

    months = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for month in months:
        file_path = immigration_data_path + 'i94_' + month + '16_sub.sas7bdat'
        df_spark = spark.read.format('com.github.saurfang.sas.spark').load(file_path)
        df_spark = df_spark.select(final_df_spark_columns)
        final_df_spark = final_df_spark.union(df_spark)

    # filter only air transportation
    final_df_spark = final_df_spark[final_df_spark['i94mode'] == 1]

    # aggregate data
    immigration_data_aggregated = final_df_spark.groupBy("i94yr", "i94mon", "i94port").agg(_sum("count"), avg('i94bir'))

    # convert to pandas dataframe
    pd_immigration_data_aggregated_df = immigration_data_aggregated.toPandas()

    # Add the information about city and State to create city_id
    with open('mapping_port_city_state.json', 'r') as fp:
        map_port_city = json.load(fp)

    mapping_port_city_df = pd.DataFrame(
        {'i94port': list(map_port_city.keys()), 'City, State Code': list(map_port_city.values())})

    # Create City ans State Code from the data dictionary of the I94 data
    mapping_port_city_df['City'] = mapping_port_city_df['City, State Code'].map(lambda x: str(x).split(',')[0])

    mapping_port_city_df['State Code'] = mapping_port_city_df['City, State Code'].map(
        lambda x: str(x).split(',')[1] if len(str(x).split(',')) > 1 else pd.np.NaN)
    mapping_port_city_df['State Code'] = mapping_port_city_df['State Code'].str.strip()
    mapping_port_city_df['State Code'] = mapping_port_city_df['State Code'].map(lambda x: str(x).split(' ')[0])

    pd_immigration_data_aggregated_df_final = pd.merge(pd_immigration_data_aggregated_df,
                                                       mapping_port_city_df[['i94port', 'City', 'State Code']],
                                                       on='i94port',
                                                       how='left')

    pd_immigration_data_aggregated_df_final['City'] = pd_immigration_data_aggregated_df_final['City'].str.lower()

    pd_immigration_data_aggregated_df_final['city_id'] = pd_immigration_data_aggregated_df_final[
        ['City', 'State Code']].apply(lambda x: hash(tuple(x)), axis=1)

    # Create primary key
    pd_immigration_data_aggregated_df_final.index = np.arange(1, len(pd_immigration_data_aggregated_df_final) + 1)
    pd_immigration_data_aggregated_df_final = pd_immigration_data_aggregated_df_final.reset_index()

    # rename the columns
    pd_immigration_data_aggregated_df_final = pd_immigration_data_aggregated_df_final.rename(columns={'i94yr': 'Year',
                                                                                                      'i94mon': 'Month',
                                                                                                      'sum(count)': 'Nb_passengers',
                                                                                                      'avg(i94bir)': 'Average_age',
                                                                                                      'index': 'immigration_id'})

    pd_immigration_data_aggregated_df_final = pd_immigration_data_aggregated_df_final[
        ['immigration_id', 'city_id', 'City',
         'Year', 'Month',
         'Nb_passengers', 'Average_age']]

    if save_local:
        pd_immigration_data_aggregated_df_final.to_csv('immigration_table.csv', sep=';', index=False)

    if save_to_s3:
        with fs.open(output_path + 'immigration_table.csv', 'wb') as output_file:
            pd_immigration_data_aggregated_df_final.to_csv(output_file, sep=';', index=False)

    return pd_immigration_data_aggregated_df_final


def main():
    """
    - Runs the full ETL pipeline
    - Cleans the temperature, demographics and immigration data, processes them and saves them on s3 or locally
    """
    temperature_data_path = '../../data2/GlobalLandTemperaturesByCity.csv'
    us_cities_demographics_data_path = 'us-cities-demographics.csv'
    immigration_data_path = '../../data/18-83510-I94-Data-2016/'

    temperature_table = clean_and_process_temperature_data(temperature_data_path)
    cities_table = clean_and_process_us_cities_demographics_data(us_cities_demographics_data_path)
    immigration_table = clean_and_process_immigration_data(immigration_data_path)
    
    return temperature_table, cities_table, immigration_table
    
    
if __name__ == "__main__":
    main()
    