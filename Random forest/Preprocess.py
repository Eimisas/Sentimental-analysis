import pandas as pd

#Reads CVS file as pandas
def readDataPandas():
    reader = pd.read_csv('data/airlinequality.csv')

    return reader

#Removes unused columns
def chooseTheAttributes(dataset):

    new_data = dataset.drop(['link', 'title',
                             'date','experience_airport',
                             'date_visit','type_traveller',
                             'terminal_cleanliness_rating',
                             'terminal_seating_rating',
                             'terminal_signs_rating',
                             'food_beverages_rating',
                             'wifi_connectivity_rating',
                             'airport_staff_rating'], axis=1)

    return new_data


def getData():

    dataset = readDataPandas();
    dataset= chooseTheAttributes(dataset)

    return dataset