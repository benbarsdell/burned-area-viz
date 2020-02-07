# Copyright (c) 2020, Ben Barsdell. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Compress output gifs using this command:
$ gifsicle -i burned_area_output.gif -O3 --colors 256 --color-method blend-diversity -o burned_area_output_shrunk.gif

Combine gifs into one using this command:
$ gifsicle g1.gif g2.gif g3.gif > combo.gif
'''

from __future__ import print_function

import sys
import os
from pyhdf.SD import SD
from PIL import Image
Image.MAX_IMAGE_PIXELS = 233280000 # Allow NASA-sized images.
from PIL import ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import imageio
import requests
import io
import getpass
import zipfile
from html.parser import HTMLParser
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta

class EquirectangularProjection(object):
    def __init__(self, lat0=0., lon0=0.):
        self._lat0 = lat0
        self._lon0 = lon0

    def from_geographic(self, lat, lon):
        '''Converts the given geographic coordinates into equirectangular in the
        range [0, 1] (north to south, west to east).
        '''
        x = ((lon - self._lon0) * np.cos(self._lat0) / np.pi + 1.) * 0.5 % 1.
        y = ((lat - self._lat0) / (np.pi / 2.) + 1.) * 0.5
        return y, x

# See http://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.2.pdf
class MODISSinusoidal500mProjection(object):
    # The radius of the idealized sphere representing the Earth in meters.
    R = 6371007.181
    # The height and width of each MODIS tile in the projection plane in meters.
    T = 1111950.
    # The western limit of the projection plane in meters.
    xmin = -20015109.
    # The northern limit of the projection plane in meters.
    ymax = 10007555.
    # The actual size of a “500-m” MODIS sinusoidal grid cell in meters.
    w = T / 2400.

    def from_geographic(self, lat, lon):
        '''Converts geographic coordinates to MODIS tile and grid cell
        coordinates (for 500m resolution MODIS datasets).

        Parameters:
        lat: Earth latitude in radians,
        lon: Earth longitude in radians.

        Returns:
        H: horizontal MODIS tile in range [0, 36),
        V: vertical MODIS tile in range [0, 18),
        i: row of grid cell within tile in range [0, 2400),
        j: column of grid cell within tile in range [0, 2400).
        '''
        x = self.R * lon * np.cos(lat)
        y = self.R * lat
        H = ((x - self.xmin) / self.T).astype(np.int32)
        V = ((self.ymax - y) / self.T).astype(np.int32)
        i = (np.fmod((self.ymax - y), self.T) / self.w - 0.5).astype(np.int32)
        j = (np.fmod((x - self.xmin), self.T) / self.w - 0.5).astype(np.int32)
        return H, V, i, j

    def to_geographic(self, H, V, i, j):
        '''Converts MODIS tile and grid cell coordinates to geographic
        coordinates (for 500m resolution MODIS datasets).

        Parameters:
        H: horizontal MODIS tile in range [0, 36),
        V: vertical MODIS tile in range [0, 18),
        i: row of grid cell within tile in range [0, 2400),
        j: column of grid cell within tile in range [0, 2400).

        Returns:
        lat: Earth latitude in radians,
        lon: Earth longitude in radians.
        '''
        x = (j + 0.5) * self.w + H * self.T + self.xmin
        y = self.ymax - (i + 0.5) * self.w - V * self.T
        lat = y / self.R
        lon = x / (self.R * np.cos(lat))
        return lat, lon

class LambertConformalProjection(object):
    def __init__(self, lat0, lon0, lat1=-0.314159, lat2=-0.628319):
        self._lon0 = lon0
        n = (np.log(np.cos(lat1) / np.cos(lat2)) /
             np.log(np.tan(np.pi / 4. + lat2 / 2.) /
                    np.tan(np.pi / 4. + lat1 / 2.)))
        self._n = n
        self._F = np.cos(lat1) * np.tan(np.pi / 4. + lat1 / 2.)**n / n
        self._rho0 = self._F / np.tan(np.pi / 4. + lat0 / 2.)**n

    def to_geographic(self, y, x):
        '''Returns latitude and longitude for the given northing and easting.'''
        rho = np.sign(self._n) * np.sqrt(x**2 + (self._rho0 - y)**2)
        theta = np.arctan(x / (self._rho0 - y))
        lat = 2 * np.arctan((self._F / rho)**(1. / self._n)) - np.pi / 2.
        lon = self._lon0 + theta / self._n
        return lat, lon

class AlbersProjection(object):
    def __init__(self, lat0, lon0, lat1, lat2):
        self._n = 0.5 * (np.sin(lat1) + np.sin(lat2))
        self._C = np.cos(lat1)**2 + 2 * self._n * np.sin(lat1)
        self._rho0 = self._rho(lat0)
        self._lon0 = lon0

    def _rho(self, lat):
        return np.sqrt(self._C - 2 * self._n * np.sin(lat)) / self._n

    def from_geographic(self, lat, lon):
        theta = self._n * (lon - self._lon0)
        rho = self._rho(lat)
        x = rho * np.sin(theta)
        y = self._rho0 - rho * np.cos(theta)
        return y, x

class AustraliaPopulationGridProjection(object):
    def __init__(self):
        self._albers_projection = AlbersProjection(
            np.radians(0.), np.radians(132.),
            np.radians(-18.), np.radians(-36.))

    def from_geographic(self, lat, lon):
        ay, ax = self._albers_projection.from_geographic(lat, lon)
        R = 6378137.0
        ay *= R
        ax *= R
        # These were manually extracted from the xml file accompanying the data.
        xmin, xmax = -3901000., 3500000.
        ymin, ymax = -5100000., -1000000.
        # This was derived by manual observation to align the maps. I think the
        # need for it stems from using a spherical instead of ellipsoidal
        # projection model (i.e., we aren't taking into account the Earth's
        # flattening here).
        fudge = 1. / 0.9955
        ymin *= fudge
        au = (1. - (ay - ymin) / (ymax - ymin))
        av = (ax - xmin) / (xmax - xmin)
        return au, av

_days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
def days_in_month(month, is_leap_year=False):
    '''Returns the number of days in the given 1-based month.'''
    n = _days_in_month[month - 1]
    if is_leap_year and month == 2:
        n += 1
    return n

_month_to_day_of_year = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
def month_to_day_of_year(month, is_leap_year=False):
    '''Returns the 1-based day of the year corresponding to the first day of the
       given 1-based month number.
    '''
    day_of_year = _month_to_day_of_year[month - 1]
    if is_leap_year and month > 2:
        day_of_year += 1
    return day_of_year

_day_of_year_to_month = ([1] * 31 + [2] * 28 + [3] * 31 + [4] * 30 + [5] * 31 +
                         [6] * 30 + [7] * 31 + [8] * 31 + [9] * 30 + [10]*31 +
                         [11] * 30 + [12] * 31)
def day_of_year_to_month(day_of_year, is_leap_year=False):
    '''Returns the 1-based month number corresponding to the given 1-based day
       of the year.
    '''
    if is_leap_year and day_of_year >= 31 + 29:
        day_of_year -= 1
    return _day_of_year_to_month[day_of_year - 1]

def interpolate_colors(rgb0, rgb1, factor):
    hsv0 = matplotlib.colors.rgb_to_hsv(rgb0)
    hsv1 = matplotlib.colors.rgb_to_hsv(rgb1)
    hsv = (1. - factor) * hsv0 + factor * hsv1
    return matplotlib.colors.hsv_to_rgb(hsv).astype(np.uint8)

_nasa_blue_marble_ng_monthly_image_url_codes = [
    '73000/73938', '73000/73967', '73000/73992', '74000/74017',
    '74000/74042', '76000/76487', '74000/74092', '74000/74117',
    '74000/74142', '74000/74167', '74000/74192', '74000/74218']
def get_blue_marble_image(month):
    filename = 'world.2004%02i.3x21600x10800.jpg' % month
    fullpath = os.path.join(os.path.expanduser('~'),
                            '.cache/burned_area/' + filename)
    if os.path.exists(fullpath):
        with open(fullpath, 'rb') as f:
            imgdata = f.read()
    else:
        code = _nasa_blue_marble_ng_monthly_image_url_codes[month - 1]
        url = ('https://eoimages.gsfc.nasa.gov/images/imagerecords/' +
               '%s/' + filename) % (code)
        print("Downloading", url)
        req = requests.get(url)
        req.raise_for_status()
        imgdata = req.content
        os.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with open(fullpath, 'wb') as f:
            f.write(imgdata)
    img = Image.open(io.BytesIO(imgdata))
    return img

def interpolate_within_month(img0, img1, month, day, is_leap_year=False):
    factor = ((day - 1) / float(days_in_month(month, is_leap_year)))
    return interpolate_colors(img0, img1, factor)

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if ((original_parsed.hostname != redirect_parsed.hostname) and
                redirect_parsed.hostname != self.AUTH_HOST and
                original_parsed.hostname != self.AUTH_HOST):
                del headers['Authorization']

# Global session instance to persist auth between downloads.
# Special session required for auth to work on MODIS data server.
g_requests_session = SessionWithHeaderRedirection()

class ModisMCD64A1Downloader(object):
    def __init__(self, year, month):
        base_url = (
            'https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.006/%04i.%02i.01' %
            (year, month))
        self._sess = g_requests_session
        cache_filename = os.path.basename(base_url)
        cache_fullpath = os.path.join(os.path.expanduser('~'),
                                      '.cache/burned_area/' + cache_filename)
        if os.path.exists(cache_fullpath):
            with open(cache_fullpath, 'r') as f:
                html = f.read()
        else:
            print("Downloading", base_url)
            html = self._get_url(base_url).text
            os.makedirs(os.path.dirname(cache_fullpath), exist_ok=True)
            with open(cache_fullpath, 'w') as f:
                f.write(html)

        hdf_filenames = []
        class HdfLinkParser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'a':
                    link = attrs['href']
                    if link.endswith('.hdf'):
                        hdf_filenames.append(os.path.join(base_url, link))
        parser = HdfLinkParser()
        parser.feed(html)
        day_of_year = month_to_day_of_year(month, is_leap_year=year % 4 == 0)
        self._stem_url = os.path.join(base_url,
                                      'MCD64A1.A%04i%03i' % (year, day_of_year))
        prefix_len = len(self._stem_url + '.h10v05.006.')
        filename_prefix_map = {}
        for f in hdf_filenames:
            file_prefix = f[:prefix_len]
            if file_prefix in filename_prefix_map:
                raise KeyError("Multiple files found with prefix %s" %
                               file_prefix)
            filename_prefix_map[file_prefix] = f
        self._filename_prefix_map = filename_prefix_map

    def _get_login(self):
        print('--------------------')
        print("NASA Earthdata Login")
        print('--------------------')
        print("To obtain a free account, visit: "
              "https://urs.earthdata.nasa.gov/users/new")
        username = input('Username: ')
        password = getpass.getpass()
        return username, password

    def _get_url(self, url):
        req = self._sess.get(url)
        if (req.status_code == requests.codes.forbidden or
            req.status_code == requests.codes.unauthorized):
            self._sess.auth = self._get_login()
            req = self._sess.get(url)
        if req.status_code != requests.codes.ok:
            print("Error:", req.text)
        req.raise_for_status()
        return req

    def __call__(self, tile_h, tile_v):
        prefix = self._stem_url + '.h%02iv%02i.006.' % (tile_h, tile_v)
        if prefix not in self._filename_prefix_map:
            raise KeyError("No file found with prefix %s" % prefix)
        filename = self._filename_prefix_map[prefix]
        cache_filename = os.path.basename(filename)
        cache_fullpath = os.path.join(os.path.expanduser('~'),
                                      '.cache/burned_area/' + cache_filename)
        if os.path.exists(cache_fullpath):
            with open(cache_fullpath, 'rb') as f:
                hdf_data = f.read()
        else:
            print("Downloading", filename)
            hdf_data = self._get_url(filename).content
            os.makedirs(os.path.dirname(cache_fullpath), exist_ok=True)
            with open(cache_fullpath, 'wb') as f:
                f.write(hdf_data)
        return hdf_data

class ModisMCD64A1Dataset(object):
    def __init__(self, year, month):
        # Note: Due to the way calloc works, this is lazily allocated, so only
        # the parts that we touch require memory (i.e., it doesn't actually use
        # up 7.5 GB).
        self._data = np.zeros([18, 36, 2400, 2400], np.int16)
        self._downloader = ModisMCD64A1Downloader(year, month)
        self._tile_set = set()

    def _extract_hdf_data(self, hdf_data, dataset_name):
        # PyHDF doesn't seem to support loading files from memory.
        with NamedTemporaryFile(delete=False) as hdf_file:
            hdf_file.write(hdf_data)
        f = SD(hdf_file.name)
        data_ds = f.select(dataset_name)
        data = data_ds[:, :]
        os.unlink(hdf_file.name)
        return data

    def _download_tile(self, tile_h, tile_v):
        try:
            hdf_data = self._downloader(tile_h, tile_v)
        except KeyError:
            print("WARNING: Tile not found:", tile_h, tile_v)
            return # Leave missing tiles as zeros.
        self._data[tile_v, tile_h] = self._extract_hdf_data(hdf_data,
                                                            'Burn Date')

    def __getitem__(self, key):
        tile_v, tile_h = key[:2]
        if len(key) > 2:
            grid_i, grid_j = key[2:]
        tile_h = np.asarray(tile_h)
        tile_v = np.asarray(tile_v)
        for h, v in zip(tile_h.flatten(), tile_v.flatten()):
            tile_key = h, v
            if tile_key not in self._tile_set:
                self._download_tile(h, v)
                self._tile_set.add(tile_key)
        if len(key) == 2:
            return self._data[tile_v, tile_h]
        return self._data[tile_v, tile_h, grid_i, grid_j]

    def get_touched_tiles(self):
        return self._tile_set

burn_color_table_rgb = np.array([
    [0x30, 0xC0, 0xFF], # Water
    [0xFF, 0x00, 0xFF], # Missing data
    [0x00, 0x80, 0x00], # Unburned
    [0xFF, 0xFF, 0xFF], # Burn day
    [0xFF, 0x80, 0x00], # Burn day + 1
    [0xFF, 0x20, 0x00], # Burn day + 2
    [0xC0, 0x00, 0x00], # Burn day + 3
    [0x80, 0x00, 0x00], # Burn day + 4+
], dtype=np.uint8)

cell_scale_m = 463.31271653 # metres
cell_area_ha = cell_scale_m**2 / 1e4 # hectares
cell_area_sqkm = cell_scale_m**2 / 1e6 # square kilometres

def bilinear_interpolate(data, y, x):
    yi = y.astype(np.int32)
    xi = x.astype(np.int32)
    dy1 = y - yi
    dy0 = 1 - dy1
    dx1 = x - xi
    dx0 = 1 - dx1
    def take(data, i, j):
        # Clip values to border to avoid out-of-bounds accesses.
        i[i < 0] = 0
        i[i >= data.shape[0]] = data.shape[0] - 1
        j[j < 0] = 0
        j[j >= data.shape[1]] = data.shape[1] - 1
        return data[i, j]
    return (take(data, yi + 0, xi + 0) * dy0 * dx0 +
            take(data, yi + 1, xi + 0) * dy1 * dx0 +
            take(data, yi + 0, xi + 1) * dy0 * dx1 +
            take(data, yi + 1, xi + 1) * dy1 * dx1)

def get_australian_population_grid():
    filename = 'apg17e_r_001.tif'
    fullpath = os.path.join(os.path.expanduser('~'),
                            '.cache/burned_area/' + filename)
    if os.path.exists(fullpath):
        with open(fullpath, 'rb') as f:
            imgdata = f.read()
    else:
        url = "https://www.ausstats.abs.gov.au/ausstats/subscriber.nsf/0/29FBBC329393CEC3CA2582FA0017B119/$File/australian%20population%20grid%202017%20in%20tiff%20format.zip"
        print("Downloading", url)
        req = requests.get(url)
        req.raise_for_status()
        zipdata = req.content
        z = zipfile.ZipFile(io.BytesIO(zipdata))
        with z.open(filename) as f:
            imgdata = f.read()
        os.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with open(fullpath, 'wb') as f:
            f.write(imgdata)
    img = Image.open(io.BytesIO(imgdata))
    return img

def main():
    output_filename = 'burned_area_output.gif'
    start_date = datetime(2019, 8, 1)
    num_days = 31 + 30 + 31 + 30 + 31
    #num_days = 1 # HACK TESTING
    #height, width = 882, 882
    height, width = 832, 832
    if len(sys.argv) <= 1:
        print("Usage:", sys.argv[0],
              ("[start_date=%04i-%02i-%02i] [num_days=%i] [output_filename=%s] "
              "[resolution=%i]") %
              (start_date.year, start_date.month, start_date.day, num_days,
               output_filename, height))
    if len(sys.argv) > 1:
        start_date = datetime(*[int(x) for x in sys.argv[1].split('-')])
    if len(sys.argv) > 2:
        num_days = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_filename = sys.argv[3]
    if len(sys.argv) > 4:
        height = width = int(sys.argv[4])

    # Pre-download all of the Blue Marble images to avoid auth timeouts below.
    for month in range(1, 13):
        get_blue_marble_image(month)

    projection = LambertConformalProjection(
        np.radians(-28.), np.radians(133.2), np.radians(-18.), np.radians(-36.))
    # Invert y axis because image pixels increase southward unlike latitude.
    grid_x = (np.arange(width)[None, :] / float(height) - 0.5) * 2. * 0.35 * 0.9
    grid_y = (
        np.arange(height)[:, None] / float(height) - 0.5) * -2. * 0.35 * 0.9

    # Zoom to SE of the country.
    grid_x = grid_x * 0.5 + 0.175 + 15 / 1024.
    grid_y = grid_y * 0.5 - 0.14
    km_per_pxl = 1.9775 * 1024. / width # Measured manually

    lat, lon = projection.to_geographic(grid_y, grid_x)
    #H, V, p, q = geographic_to_modis_sinusoidal_500m(lat, lon)
    modis_projection = MODISSinusoidal500mProjection()
    H, V, p, q = modis_projection.from_geographic(lat, lon)
    y, x = EquirectangularProjection().from_geographic(lat, lon)
    u, v = None, None

    popden_img = get_australian_population_grid()
    popden_data = np.array(popden_img)
    nodata_value = -3.4028231e+038
    popden_data[popden_data == nodata_value] = 0
    pop_data = popden_data.copy() # Original population data.
    popden_data = np.log10(popden_data)
    popden_min = -3.
    popden_max = 4.
    popden_data = np.clip((popden_data - popden_min) / (popden_max - popden_min), 0., 1.)

    popden_projection = AustraliaPopulationGridProjection()
    def get_popden(data, lat, lon):
        au, av = popden_projection.from_geographic(lat, lon)
        return bilinear_interpolate(data, au * popden_img.height,
                                    av * popden_img.width)
    popden = get_popden(popden_data, lat, lon)
    popden = popden[..., None] # Add color dimension

    last_month = None
    burn_data = None
    burn_area_total_ha = 0.
    burn_pop_total = 0.
    quantizer = 2
    tile_pop_cache = {}
    with imageio.get_writer(output_filename, mode='I', fps=10,
                            palettesize=256, quantizer=quantizer) as writer:
        for i, date in ((n, start_date + timedelta(n))
                        for n in range(num_days)):
            year = date.year
            month = date.month
            day_of_year = date.timetuple().tm_yday
            is_leap_year = year % 4 == 0
            print('Generating frame %03i / %03i, %04i-%02i-%02i' %
                  (i + 1, num_days, year, month, date.day))

            print("Getting burn data")
            if month != last_month:
                raw_burn_data = ModisMCD64A1Dataset(year, month)
                last_month = month
                new_burn_data = raw_burn_data[V, H, p, q]
                if burn_data is None:
                    burn_data = new_burn_data
                    burn_tile_set = raw_burn_data.get_touched_tiles()
                else:
                    burn_data = np.maximum(burn_data, new_burn_data)

                print("Getting Blue Marble data")
                raw_marble_data0 = get_blue_marble_image(month)
                next_month = month % 12 + 1
                raw_marble_data1 = get_blue_marble_image(next_month)
                d = 2
                print("Resizing Blue Marble data")
                raw_marble_data0 = raw_marble_data0.resize(
                    (raw_marble_data0.width // d, raw_marble_data0.height // d),
                    Image.ANTIALIAS)
                raw_marble_data1 = raw_marble_data1.resize(
                    (raw_marble_data1.width // d, raw_marble_data1.height // d),
                    Image.ANTIALIAS)
                if u is None:
                    u = (-y * raw_marble_data0.height).astype(np.int32)
                    v = (x * raw_marble_data0.width).astype(np.int32)
                print("Projecting Blue Marble data")
                marble_data0 = np.asarray(raw_marble_data0)[u, v]
                marble_data1 = np.asarray(raw_marble_data1)[u, v]

            burn_area_today_ha = 0.
            burn_pop_today = 0.
            tile_pop_total = 0.
            for tile_h, tile_v in burn_tile_set:
                if tile_v < 10:
                    # HACK to exclude tiles North of Australia. Note that parts
                    # of NZ are included, but they don't burn.
                    # TODO: Need a political map (ideally in sinusoidal
                    # projection) to be able to properly isolate countries.
                    continue
                raw_burn_tile = raw_burn_data[tile_v, tile_h]
                cells_burned_today = raw_burn_tile == day_of_year
                burn_area_today_ha += cells_burned_today.sum() * cell_area_ha

                modis_i = np.arange(raw_burn_tile.shape[0])[None, :]
                modis_j = np.arange(raw_burn_tile.shape[1])[:, None]
                lat, lon = modis_projection.to_geographic(tile_h, tile_v,
                                                          modis_i, modis_j)
                tile_key = tile_h, tile_v
                if tile_key not in tile_pop_cache:
                    tile_popden = get_popden(pop_data, lat, lon)
                    tile_pop = tile_popden * cell_area_sqkm
                    tile_pop_cache[tile_key] = tile_pop
                tile_pop = tile_pop_cache[tile_key]
                tile_pop_total += tile_pop.sum()
                burn_pop_today += (tile_pop * cells_burned_today).sum()
            burn_pop_total += burn_pop_today
            burn_area_total_ha += burn_area_today_ha
            print("Burned area today:", burn_area_today_ha / 1e3, "total:",
                  burn_area_total_ha / 1e3)

            print("Interpolating Blue Marble data")
            marble_data = interpolate_within_month(
                marble_data0, marble_data1, month, date.day, is_leap_year)

            print("Generating composite image")
            color_key = 2 + np.where(
                burn_data <= 0, burn_data,
                1 + np.clip(day_of_year - burn_data, -1, 4))
            pop_color = [50, 200, 255]
            popburn_color = [255, 100, 255]
            background = marble_data
            frame_data = np.where((color_key <= 2)[..., None],
                                  background, burn_color_table_rgb[color_key])
            draw_pop_color = np.where(color_key[..., None] <= 2, pop_color,
                                      popburn_color)
            frame_data = ((1. - popden) * frame_data +
                          popden * draw_pop_color).astype(np.uint8)

            frame_img = Image.fromarray(frame_data)
            draw = ImageDraw.Draw(frame_img)
            font_size = 18
            font = ImageFont.truetype("FreeSans.ttf", font_size)
            font_color = (255, 255, 255)
            date_text = '%04i-%02i-%02i' % (year, month, date.day)
            dx = dy = 10
            th = 30
            row = 5
            draw.text((dx, height - row * th - dy), date_text, font_color,
                      font=font)
            row -= 1
            draw.ellipse((dx, height - row * th - dy,
                          dx + font_size, height - row * th - dy + font_size),
                         fill=tuple(pop_color))
            draw.text((dx + font_size + 5, height - row * th - dy),
                      "Population density (logarithmic scale)", font_color,
                      font=font)
            row -= 1
            draw.ellipse((dx, height - row * th - dy,
                          dx + font_size, height - row * th - dy + font_size),
                         fill=tuple(
                             interpolate_colors(burn_color_table_rgb[3],
                                                burn_color_table_rgb[4], 0.25)))
            draw.text((dx + font_size + 5, height - row * th - dy),
                      "Fire front", font_color, font=font)
            row -= 1
            draw.ellipse((dx, height - row * th - dy,
                          dx + font_size, height - row * th - dy + font_size),
                         fill=tuple(burn_color_table_rgb[7]))
            draw.text((dx + font_size + 5, height - row * th - dy),
                      "Burned area", font_color, font=font)
            row -= 1
            draw.ellipse((dx, height - row * th - dy,
                          dx + font_size, height - row * th - dy + font_size),
                         fill=tuple(popburn_color))
            draw.text((dx + font_size + 5, height - row * th - dy),
                      "Directly affected population", font_color, font=font)
            sx = width / 1024.
            sy = height / 1024.
            ox = -47
            x0, y0, x1 = (781 + ox)*sx, 444*sx, (810 + ox)*sx
            draw.line((x0 + 5, y0, x1, y0))
            draw.text((x1 + 5, y0 - font_size // 2), "Sydney", font_color,
                      font=font)
            x0, y0, x1 = (677 + ox)*sx, 509*sx, (775 + ox)*sx
            draw.line((x0 + 5, y0, x1, y0))
            draw.text((x1 + 5, y0 - font_size // 2), "Canberra", font_color,
                      font=font)
            x0, y0, x1, y1 = (473 + ox)*sx, 636*sx, (486 + ox)*sx, 730*sx
            draw.line((x0, y0 + 5, x0, y1))
            draw.line((x0, y1, x1, y1))
            draw.text((x1 + 5, y1 - font_size // 2), "Melbourne", font_color,
                      font=font)
            x0, y0, x1 = (547 + ox)*sx, 936*sx, (590 + ox)*sx
            draw.line((x0 + 5, y0, x1, y0))
            draw.text((x1 + 5, y0 - font_size // 2), "Hobart", font_color,
                      font=font)
            x0, y0, x1, y1 = (193 + ox)*sx, 449*sx, (180 + ox)*sx, (519 + 20)*sx
            draw.line((x0, y0 + 5, x0, y1))
            draw.line((x0, y1, x1, y1))
            draw.text((x1 - 70 - 5, y1 - font_size // 2), "Adelaide",
                      font_color, font=font)
            x0, y0, x1 = (927 + ox)*sx, 105*sx, (957 + ox)*sx
            draw.line((x0 + 5, y0, x1, y0))
            draw.text((x1 + 5, y0 - font_size // 2), "Brisbane", font_color,
                      font=font)

            #small_font_size = 14
            small_font_size = 11
            small_font = ImageFont.truetype("FreeSans.ttf", small_font_size)
            x0 = width - dx
            y0 = height - dy - small_font_size - 10
            km_per_mile = 1.61
            draw.line((x0 - 100. * km_per_mile / km_per_pxl, y0, x0, y0))
            draw.line((x0 - 100. / km_per_pxl, y0,
                       x0 - 100. / km_per_pxl, y0 - 5))
            draw.line((x0, y0 - 5, x0, y0 + 5))
            draw.line((x0 - 100. * km_per_mile / km_per_pxl, y0,
                       x0 - 100. * km_per_mile / km_per_pxl, y0 + 5))
            draw.text((x0 - 100. / km_per_pxl + 2, y0 - font_size + 3),
                      "100 km", font_color, font=small_font)
            draw.text((x0 - 122. / km_per_pxl + 1, y0 + 4),
                      "100 mi", font_color, font=small_font)
            frame_data = np.asarray(frame_img)

            print("Writing frame")
            writer.append_data(frame_data)

            if (i + 1 == num_days):
                # Linger on the final frame for a bit.
                for _ in range(30):
                    writer.append_data(frame_data)

if __name__ == '__main__':
    main()
