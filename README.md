<div align="center">
  <img src="https://user-images.githubusercontent.com/3979096/74024885-de627b80-49f6-11ea-9d08-6139a9304070.gif"
       alt="Australian Bush Fire Burned Area Visualisation">
</div>

# Australian Bush Fire Burned Area Visualisation

A visualisation of burned area satellite data for Australian bush fires.

Inspired by [this BBC article](https://www.bbc.com/news/world-australia-50951043) about the [2019/2020 bush fire season](https://en.wikipedia.org/wiki/2019%E2%80%9320_Australian_bushfire_season).

Note that the burned area data are only available ~2 months after the first burn date (e.g., December data are only available in early February).

## Data sources

  * [Collection 6 MODIS Burned Area Product](https://modis.gsfc.nasa.gov/data/dataprod/mod45.php) ([data](https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.006/), [guide](http://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.2.pdf))
  * [NASA Blue Marble Collection](https://visibleearth.nasa.gov/images/57723/the-blue-marble) ([data](https://visibleearth.nasa.gov/collection/1484/blue-marble?page=1))
  * [Australian Population Grid 2017](https://www.abs.gov.au/AUSSTATS/abs@.nsf/Previousproducts/3218.0Main%20Features702016-17?opendocument&tabname=Summary&prodno=3218.0&issue=2016-17&num=&view=) ([data](https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.02016-17?OpenDocument))

## Instructions

    $ pip install -r requirements.txt
    $ python burned_area_viz.py
    $ gifsicle -i burned_area_output.gif -O3 --colors 256 --color-method blend-diversity -o burned_area_output_shrunk.gif
