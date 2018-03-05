# Generate random addresses from random cities in US
import requests
import numpy
import json
from pprint import pprint
from random import randint
from time import sleep

NOMINATIM_API_URL = 'https://nominatim.openstreetmap.org/reverse?format=json&lat={}&lon={}&zoom=18'
NUMBER_OF_ADDR = 500
DISTANCE = 100 * 1000
API_CALL_BREAK = 2

def generate_random_point(x0, y0, distance):
    """x0 - longitude, y0 - latitude"""   
    r = distance/ 111300
    u = numpy.random.uniform(0, 1)
    v = numpy.random.uniform(0, 1)
    w = r * numpy.sqrt(u)
    t = 2 * numpy.pi * v
    x = w * numpy.cos(t)
    x1 = x / numpy.cos(y0)
    y = w * numpy.sin(t)
    return (x0 + x1, y0 + y)

def reverse_geocode(longitude, latitude):
    response = requests.get(url.format(latitude, longitude), headers=headers)
    data = ''
    if response.status_code == 200:
        data = json.loads(response.text)
        data = data.get('display_name')
    return data

with open("cities.json") as fp:
    cities = json.load(fp)
    number_of_cities = len(cities)
    for _ in range(NUMBER_OF_ADDR):
        i = randint(0, number_of_cities - 1)
        city = cities[i]
        display_name = reverse_geocode(
            *generate_random_point(city['longitude'], city['latitude'], DISTANCE))
        print(display_name)
        # make API call at least once in two seconds
        sleep(API_CALL_BREAK)
