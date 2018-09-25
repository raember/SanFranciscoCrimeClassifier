#!/bin/env python3

import csv
import pytz

from datetime import datetime

import numpy as np
import tensorflow as tf
import pandas as pd
import logging as log
import os


class CsvFile:
    filename = ''
    df = None
    df_orig = None
    log = None
    COLS = ['Id', 'Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    DISTRICTS = ['BAYVIEW', 'NORTHERN', 'INGLESIDE', 'TARAVAL', 'MISSION', 'CENTRAL', 'TENDERLOIN', 'RICHMOND',
                 'SOUTHERN', 'PARK']
    min_date = pd.Timestamp('1/1/1970 00:00:00')
    sf_tz = pytz.timezone('US/Pacific')

    def __init__(self, filename, csvfile=None):
        self.log = log.getLogger(self.__class__.__name__)
        self.filename = filename
        file = self.filename + '.csv'
        self.log.debug("Reading csv file {}".format(file))
        if csvfile is not None:
            self.df_orig = csvfile.df_orig
        else:
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

    def _prepare_date(self, date):
        # date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        # date = date.astimezone(self.sf_tz)
        delta = date - self.min_date
        return int(delta.total_seconds())

    def _prepare_day(self, daystr):
        return self.DAYS.index(daystr)

    def _prepare_district(self, daystr):
        return self.DISTRICTS.index(daystr)

    @staticmethod
    def _prepare_address(addressstr):
        return hash(addressstr)

    @staticmethod
    def _prepare_latitude(latitudestr):
        return float(latitudestr)

    @staticmethod
    def _prepare_longitude(longitudestr):
        return float(longitudestr)

    def save(self):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        file = self._prep_file()
        self.log.debug("Writing csv file {}".format(file))
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
        self.log.debug("Reading csv file {}".format(file))
        self.df = self._read_prep_file(file)
        self.log.info("Read data frame of shape {}".format(self.df_orig.shape))

    def get(self, index):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        date = datetime.fromtimestamp(int(self.df.at[index, 'Dates']))
        date = date.astimezone(pytz.utc)
        # date = datetime.strptime(self.df_old.at[index, 'Dates'], '%Y-%m-%d %H:%M:%S')
        day = self.DAYS[self.df.at[index, 'DayOfWeek']]
        # day = self.df_old.at[index, 'DayOfWeek']
        district = self.DISTRICTS[self.df.at[index, 'PdDistrict']]
        # district = self.df_old.at[index, 'DayOfWeek']
        address = self.df_orig.at[index, 'Address']
        latitude = float(self.df.at[index, 'Y'])
        longitude = float(self.df.at[index, 'X'])
        return date, day, district, address, latitude, longitude

    def toNpArray(self):
        np.reshape(self.df.values, self.df.shape)
        self.df.values()


class TestDataCsvFile(CsvFile):

    def __init__(self):
        super().__init__("test")

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
        self.df['Dates'] = self.df['Dates'].apply(self._prepare_date)
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

    def _read_file(self, file):
        return pd.read_csv(
            file,
            keep_date_col=True,
            parse_dates=["Dates"]
        )

    def parse(self):
        self.df = self.df_orig.copy()
        self.log.debug('Parsing Dates')
        self.df['Dates'] = self.df['Dates'].apply(self._prepare_date)
        self.log.debug('Deleting Category')
        self.df = self.df.drop('Category', axis=1)
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

    CATEGORIES = ['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

    def __init__(self, csvfile=None):
        super().__init__("train", csvfile)

    def _prep_file(self):
        return self.filename + '_labels_prep.csv'

    def _read_file(self, file):
        return pd.read_csv(file)

    def _read_prep_file(self, file):
        return pd.read_csv(
            file,
            index_col='Id'
        )

    def _prepare_category(self, category):
        return self.CATEGORIES.index(category)

    def parse(self):
        self.df = self.df_orig.copy()
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
        self.log.info('Parsed dataframe')

    def get(self, index):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        district = self.CATEGORIES[self.df.at[index, 'Category']]
        return district


# TODO: Make a class for the train data.