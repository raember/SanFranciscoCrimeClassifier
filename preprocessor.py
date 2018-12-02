#!/bin/env python3

import logging as log
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz


class CsvFile:
    filename = ''
    df: pd.DataFrame = None
    df_orig: pd.DataFrame = None
    log = None
    COLS = [
        'Id',
        'Dates',
        'Year',
        'Month',
        'Day',
        'Hour',
        'Minute',
        'Weekday',
        'Season',
        'Daynight',
        'DayOfWeek',
        'PdDistrict',
        'Address',
        'X',
        'Y'
    ]
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    DISTRICTS = ['BAYVIEW', 'NORTHERN', 'INGLESIDE', 'TARAVAL', 'MISSION', 'CENTRAL', 'TENDERLOIN', 'RICHMOND',
                 'SOUTHERN', 'PARK']
    min_date = pd.Timestamp('1/1/2003 00:00:00')
    max_date = pd.Timestamp('1/1/2016 00:00:00')
    sf_tz = pytz.timezone('US/Pacific')

    def __init__(self, filename, csvfile=None):
        self.log = log.getLogger(self.__class__.__name__)
        self.filename = filename
        file = self.filename + '.csv'
        if csvfile is not None:
            self.df_orig = csvfile.df_orig
            self.log.info("Linked to data frame of '{}' with shape {}".format(csvfile.filename, self.df_orig.shape))
        else:
            self.log.debug("Reading csv file '{}'".format(file))
            self.df_orig = self._read_file(file)
            self.log.info("Read data frame of shape {}".format(self.df_orig.shape))

    def _read_file(self, file):
        raise NotImplementedError()

    def _read_prep_file(self, file):
        return pd.read_csv(
            file,
            index_col='Id',
            keep_date_col=True,
            parse_dates=["Dates"]
        )

    def parse(self):
        raise NotImplementedError()

    def _prepare_date(self, date: datetime):
        # date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        # date = date.astimezone(self.sf_tz)
        max_delta = self.max_date - self.min_date
        delta = date - self.min_date
        return int(delta.total_seconds())/max_delta.total_seconds()

    def _transform_date(self):
        def get_season(month: int):
            if month < 3 or month == 12:
                return 0  # December - February: Winter
            if month < 6:
                return 1  # March - May: Spring
            if month < 9:
                return 2  # June - August: Summer
            if month < 12:
                return 3  # September - November: Autumn
            raise Exception()

        def get_day_night(hour: int):
            if hour <= 7 or hour >= 7:
                return 0  # Night
            return 1  # Day

        self.log.debug("Parsing Year")
        self.df['Year'] = self.df['Dates'].apply(lambda x: x.year)
        self.log.debug("Parsing Month")
        self.df['Month'] = self.df['Dates'].apply(lambda x: x.month)
        self.log.debug("Parsing Day")
        self.df['Day'] = self.df['Dates'].apply(lambda x: x.day)
        self.log.debug("Parsing Hour")
        self.df['Hour'] = self.df['Dates'].apply(lambda x: x.hour)
        self.log.debug("Parsing Minute")
        self.df['Minute'] = self.df['Dates'].apply(lambda x: x.minute)
        self.log.debug("Parsing Weekday")
        self.df['Weekday'] = self.df['Dates'].apply(lambda x: x.isoweekday())
        self.log.debug("Parsing Season")
        self.df['Season'] = self.df['Dates'].apply(lambda x: get_season(x.month))
        self.log.debug("Parsing Daynight")
        self.df['Daynight'] = self.df['Dates'].apply(lambda x: get_day_night(x.day))
        self.log.debug("Parsing Dates")
        self.df['Dates'] = self.df['Dates'].apply(self._prepare_date)

    def _prepare_day(self, daystr):
        return self.DAYS.index(daystr)/(len(self.DAYS)/2)-1

    def _prepare_district(self, daystr):
        return self.DISTRICTS.index(daystr)/(len(self.DISTRICTS)/2)-1

    @staticmethod
    def _prepare_address(addressstr):
        return hash(addressstr)/(sys.maxsize/2)-1

    @staticmethod
    def _prepare_latitude(latitudestr):
        return float(latitudestr)/180

    @staticmethod
    def _prepare_longitude(longitudestr):
        return float(longitudestr)/180

    def save(self):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        file = self._prep_file()
        self.log.debug("Writing csv file '{}'".format(file))
        self.df.to_csv(
            file,
            index_label='Id'
        )
        self.log.info("Wrote data frame of shape {}".format(self.df.shape))

    def _prep_file(self):
        return self.filename + '_prep.csv'

    def prep_file_exists(self):
        return os.path.isfile(self._prep_file())

    def load(self):
        file = self._prep_file()
        self.log.debug("Reading csv file '{}'".format(file))
        self.df = self._read_prep_file(file)
        self.log.info("Read data frame of shape {}".format(self.df_orig.shape))

    def get(self, index):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        date = self.df_orig['Dates'][index]
        # date = datetime.strptime(self.df_old.at[index, 'Dates'], '%Y-%m-%d %H:%M:%S')
        day = self.df_orig['DayOfWeek'][index]
        # day = self.df_old.at[index, 'DayOfWeek']
        district = self.df_orig['PdDistrict'][index]
        # district = self.df_old.at[index, 'DayOfWeek']
        address = self.df_orig['Address'][index]
        latitude = float(self.df_orig['Y'][index])
        longitude = float(self.df_orig['X'][index])
        return date, day, district, address, latitude, longitude

    def toNpArray(self):
        return np.reshape(self.df.values, self.df.shape)


class TestDataCsvFile(CsvFile):

    def __init__(self):
        super().__init__("test")

    def _prep_file(self):
        return self.filename + '_samples.csv'

    def _read_file(self, file):
        return pd.read_csv(
            file,
            index_col='Id',
            keep_date_col=True,
            parse_dates=["Dates"]
        )

    def parse(self):
        self.df = self.df_orig.copy()
        self.log.debug('Parsing Dates')
        self._transform_date()
        self.log.debug('Parsing Day of the week')
        self.df['DayOfWeek'] = self.df['DayOfWeek'].apply(self._prepare_day)
        self.log.debug('Parsing District')
        self.df['PdDistrict'] = self.df['PdDistrict'].apply(self._prepare_district)
        self.log.debug('Parsing Address')
        self.df['Address'] = self.df['Address'].apply(self._prepare_address)
        self.log.debug('Parsing Longitude')
        self.df['X'] = self.df['X'].apply(self._prepare_longitude)
        self.log.debug('Parsing Latitude')
        self.df['Y'] = self.df['Y'].apply(self._prepare_latitude)
        self.log.info('Parsed dataframe')


class TrainDataCsvFile(CsvFile):

    def __init__(self, csvfile=None):
        super().__init__("train", csvfile)

    def _prep_file(self):
        return self.filename + '_samples.csv'

    def _read_file(self, file):
        return pd.read_csv(
            file,
            keep_date_col=True,
            parse_dates=["Dates"]
        )

    def parse(self):
        self.df = self.df_orig.copy()
        # print(self.df.shape)
        # print(self.df['Dates'].shape)
        # print(self.df[['Dates']].shape)
        # print(self.df.at[1, 'Dates'])
        # print(self.df.at[1, 'Dates'].minute)
        # exit(0)
        self.log.debug('Parsing Dates')
        self._transform_date()
        self.log.debug('Deleting Category')
        self.df = self.df.drop('Category', axis=1)
        # self.log.debug('Parsing Category')
        # self.df['Category'] = self.df['Category'].apply(self._prepare_category)
        self.log.debug('Deleting Descript')
        self.df = self.df.drop('Descript', axis=1)
        self.log.debug('Parsing Day of the week')
        self.df['DayOfWeek'] = self.df['DayOfWeek'].apply(self._prepare_day)
        self.log.debug('Parsing District')
        self.df['PdDistrict'] = self.df['PdDistrict'].apply(self._prepare_district)
        self.log.debug('Deleting Resolution')
        self.df = self.df.drop('Resolution', axis=1)
        self.log.debug('Parsing Address')
        self.df['Address'] = self.df['Address'].apply(self._prepare_address)
        self.log.debug('Parsing Longitude')
        self.df['X'] = self.df['X'].apply(self._prepare_longitude)
        self.log.debug('Parsing Latitude')
        self.df['Y'] = self.df['Y'].apply(self._prepare_latitude)
        self.log.info('Parsed dataframe')


class TrainLabelsCsvFile(CsvFile):
    CATEGORIES = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
                  'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
                  'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
                  'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
                  'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
                  'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
                  'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']
    stats = {}
    count = 0

    def __init__(self, csvfile=None):
        super().__init__("train", csvfile)

    def _prep_file(self):
        return self.filename + '_labels.csv'

    def _read_file(self, file):
        return pd.read_csv(file)

    def _read_prep_file(self, file):
        return pd.read_csv(
            file,
            index_col='Id'
        )

    def _prepare_category(self, category):
        cat = self.CATEGORIES.index(category)
        if not cat in self.stats:
            self.stats[cat] = 0.0
        self.stats[cat] += 1.0
        self.count += 1
        return cat

    def parse(self):
        self.df = self.df_orig.copy()
        # self.log.debug('Deleting Id')
        # self.df = self.df.drop('Id', axis=1)
        self.log.debug('Deleting Dates')
        self.df = self.df.drop('Dates', axis=1)
        self.log.debug('Parsing Category')
        self.df['Category'] = self.df['Category'].apply(self._prepare_category)
        self.log.debug('Deleting Descript')
        self.df = self.df.drop('Descript', axis=1)
        self.log.debug('Deleting DayOfWeek')
        self.df = self.df.drop('DayOfWeek', axis=1)
        self.log.debug('Deleting PdDistrict')
        self.df = self.df.drop('PdDistrict', axis=1)
        self.log.debug('Deleting Resolution')
        self.df = self.df.drop('Resolution', axis=1)
        self.log.debug('Deleting Address')
        self.df = self.df.drop('Address', axis=1)
        self.log.debug('Deleting Longitude')
        self.df = self.df.drop('X', axis=1)
        self.log.debug('Deleting Latitude')
        self.df = self.df.drop('Y', axis=1)
        self.log.debug('Parsed dataframe')

    def get(self, index):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        district = self.CATEGORIES[self.df.at[index, 'Category']]
        return district
