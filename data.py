# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import datetime
import os

import pandas as pd
from shapefile import Reader


def load():
    # Determine paths.
    file_dir = os.path.dirname(os.path.abspath(__file__))
    locs_path = os.path.join(file_dir, 'data', 'IDSTA.shp')
    data_path = os.path.join(file_dir, 'data', 'HRtemp2006.txt')

    # Load locations.
    sf = Reader(locs_path)
    names, lons, lats = [], [], []
    for sr in sf.shapeRecords():
        name = sr.record.as_dict()['IDT_AK']
        lon, lat = sr.shape.points[0]
        names.append(name)
        lons.append(lon)
        lats.append(lat)
    locs = pd.DataFrame({'lon': lons, 'lat': lats},
                        index=pd.Index(names, name='node'))

    # Read data.
    df = pd.read_csv(data_path, sep='\t')

    # Rename things.
    df = pd.DataFrame({'node': df['IDT_AK'],
                       'date': df['DATE'],
                       'temp': df['MDTEMP']})

    # Make columns nodes.
    df = df.set_index(['date', 'node']).unstack('node')['temp']

    # Drop outputs with missing values, which are only a few.
    df = df.dropna(axis=1)

    # Parse dates and convert to day in the year 2006.
    xs = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in df.index]
    start = datetime.datetime(year=2006, month=1, day=1)
    df['day'] = [(x - start).total_seconds() / 3600 / 24 + 1 for x in xs]
    df = df.set_index('day').sort_index()

    # Filter locations by kept nodes.
    locs = locs.reindex(df.columns, axis=0)

    return locs, df
