import argparse
import json
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import os

def main():

   # Get the URL and destination file
   url, destination = parseCommandLine()
   
   # Download all of the image data
   data = download_data(url)

   # Save the data to a JSON file
   with open(destination, 'w') as output_file:
      json.dump(data, output_file, indent=2)

   # Delete temporary file used in downloading
   os.remove('temp')

   print('Data for {} images has been saved to {}'.format(str(len(data)), destination))

# Retrieve the URL and optional file destination
def parseCommandLine():
   parser = argparse.ArgumentParser(description='Create a json file from an URL specifying a LabelMe dataset.')
   parser.add_argument('URL', help='location of the LabelMe dataset')
   parser.add_argument('destination', nargs='?', default='labelme.json', help='destination JSON file (default is labelme.json)')
   args = parser.parse_args()
   return args.URL, args.destination

def download_data(url):
   # Get the data from the given URL
   try:
      page = requests.get('{}/Annotations'.format(url)).text
   except:
      print('Error reading data form specified url')
      return []

   data = []

   all_xml_files = ['{}/Annotations/{}'.format(url, node.get('href')) for node in BeautifulSoup(page, 'html.parser').find_all('a') if node.get('href').endswith('xml')]
   print('Downloading data for {} images...\n'.format(len(all_xml_files)))

   # For each xml file at <URL>/Annotations
   for xml_url in all_xml_files:

      # Download the XML file
      try:
         with open('temp', 'wb') as f:
            f.write(requests.get(xml_url).content)
      except:
         print('Error downloading data from {}'.format(xml_url))
         continue

      # Get the root of the XML document
      root = ET.parse('temp').getroot()

      image_data = {}

      # Name of the image file
      image_name = root.find('filename').text.strip()
      image_data['task_id'] = image_name.split('.')[0]

      # URL of the image
      image_url = '{}/Images/{}'.format(url, image_name)
      image_data['params'] = {'attachment': image_url}

      # Approval status
      image_data['customer_review_status'] = 'accepted'

      # Polygon coordinates of image
      all_objects = root.findall('object')
      if len(all_objects) == 0: # if the XML file has no polygons
         width = int(root.find('imagesize').find('ncols').text.strip())
         height = int(root.find('imagesize').find('nrows').text.strip())
         # Create a line at the bottom of the image
         vertices = [{'x': 0, 'y': height - 1}, {'x': width, 'y': height - 1}, {'x': width, 'y': height}, {'x': 0, 'y': height}]
         image_data['response'] = {'annotations': [{'vertices': vertices}]}
      else: # get the coordinates from the XML file
         polygons = []
         for polygon in all_objects:
            # if the polygon has not been deleted
            if polygon.find('deleted').text == '0':
               vertices = []
               for point in polygon.find('polygon').findall('pt'):
                  x = point[0].text
                  y = point[1].text
                  vertices.append({'x': int(x), 'y': int(y)})
               polygons.append({'vertices': vertices})
         image_data['response'] = {'annotations': polygons}

      data.append(image_data)

   return data

if __name__ == '__main__':
   main()
