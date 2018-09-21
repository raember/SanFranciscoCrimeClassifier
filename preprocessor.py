#!/bin/env python3

import csv
import pytz

from datetime import datetime

from numpy import *
import tensorflow as tf
import pandas as pd
import logging as log
import os


class DataFile:
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

    def __init__(self, filename):
        self.log = log.getLogger(self.__class__.__name__)
        self.filename = filename
        file = self.filename + '.csv'
        self.log.debug("Reading csv file {}".format(file))
        self.df_orig = pd.read_csv(
            file,
            index_col='Id',
            keep_date_col=True,
            parse_dates=["Dates"]
        )
        self.log.info("Read data frame of shape {}".format(self.df_orig.shape))

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

    def _prepare_address(self, addressstr):
        return hash(addressstr)

    def _prepare_latitude(self, latitudestr):
        return float(latitudestr)

    def _prepare_longitude(self, longitudestr):
        return float(longitudestr)

    def save(self):
        if self.df is None:
            self.log.error("Data not yet parsed")
            return
        file = self.filename + '_prep.csv'
        self.log.debug("Writing csv file {}".format(file))
        self.df.to_csv(
            file,
            index_label='Id'
        )
        self.log.info("Wrote data frame of shape {}".format(self.df.shape))

    def prep_file_exists(self):
        file = self.filename + '_prep.csv'
        return os.path.isfile(file)

    def load(self):
        file = self.filename + '_prep.csv'
        self.log.debug("Reading csv file {}".format(file))
        self.df = pd.read_csv(
            file,
            index_col='Id',
            keep_date_col=True,
            parse_dates=["Dates"]
        )
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


class TestDataFile(DataFile):

    def __init__(self):
        super().__init__("test")

    def parse(self):
        self.df = self.df_orig.copy()
        # print(self.df['Dates'].at[0])
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
        # self.log.debug('Deleting Id')
        # self.df = self.df.drop('Id', axis=1)
        self.log.info('Parsed dataframe')


# TODO: Make a class for the train data.