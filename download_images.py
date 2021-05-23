import os
import random
import csv
import pandas as pd

################################################################################
# So this next import is interesting. Instead of using the official results
# class from the google streetview code base, we are importing a near copy
# of that class that we define in api_custom. The reason for this is that we 
# wanted to mess with the file structure when saving the query result.

# Therefore, we are no longer using:
# import google_streetview.api

# And instead using:
import api_custom
################################################################################

def get_image(lat, lng):
	"""
	Retrieves the jpg and metadata for the given latitude and longitude.
	"""

	# Define parameters for street view api
	params = [{
		'size': '640x640', # max 640x640 pixels
		'location': f'{lat},{lng}',
		'key': 'AIzaSyAPMOXydvR5OXd7wv_eEHoaSmnrUChGC2g',
		'radius': 500,
	}]

	# Create a results object
	results = api_custom.results(params)

	# Download images to directory 'data/images'
	results.download_links(f'data/images')


def random_coords(lat, lng, minutes=1):
	"""
	Takes the input coordinates and adds/subtracts random minutes from each.
	The random additions/subtractions come from a uniform distribution with
	max of the "minutes" input.

	Note: a coordinate minute is approximately 1.15 miles. A coordinate minute
	is represented by a 1/60 ~ .0166667
	"""

	lat_new = float(lat) + random.uniform(0, minutes / 60)
	lng_new = float(lng) + random.uniform(0, minutes / 60)

	return (lat_new, lng_new)


def sample_cities(size=1):
	"""
	Retrieves a random selection of cities (rows) from the city_coords.csv.
	"""

	city_df = pd.read_csv('data/city_coords.csv')
	big_city_df = city_df.loc[city_df.population >= 50000, :]

	num_cities = big_city_df.shape[0]

	rand_nums = random.sample(range(0, num_cities), size)
	selected_cities_df = big_city_df.iloc[rand_nums, :]
	selected_cities = [dict(row) for i, row in selected_cities_df.iterrows()]

	return selected_cities


def main(size=1, adjust=0):
	"""
	Single function that retrieves the images for a sample of cities.

	size: integer indicating how many cities we are collecting in our sample.
	adjust: a numeric indicating how much we should randomize each city's 
	coordinates.
	"""

	# Sample the cities
	cities = sample_cities(size=size)

	for city in cities:
		# Adjust coordinates
		lat, lng = random_coords(city['lat'], city['lng'], adjust)

		# Get the image and metadata.
		try:
			get_image(lat, lng)
			print(f"Successfully retrieved image for {city['city_ascii'], city['iso3']} at the coordinates: {lat}, {lng}")

		except ValueError:
			print(f"There was no image for {city['city_ascii'], city['iso3']} at the coordinates: {lat}, {lng}")

	return 

main(size=2)