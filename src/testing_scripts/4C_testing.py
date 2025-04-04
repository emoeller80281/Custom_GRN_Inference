import os
import json
import requests
import logging
from dcicutils import ff_utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_api_credentials(keypairs_path):
    """
    Loads API credentials from a JSON file.
    Expected format:
    {
      "default": {
          "key": "XXXXXXXX",
          "secret": "xxxxxxxxxxxxxxxx",
          "server": "https://data.4dnucleome.org"
      }
    }
    """
    with open(keypairs_path, 'r') as f:
        keypairs = json.load(f)
    default_key = keypairs["default"]["key"]
    default_secret = keypairs["default"]["secret"]
    server = keypairs["default"]["server"]
    return default_key, default_secret, server


# Path to your keypairs.json file (adjust if necessary)
keypairs_path = os.path.expanduser("~/keypairs.json")
default_key, default_secret, server = load_api_credentials(keypairs_path)

key = {'key': default_key, 'secret': default_secret, 'server': 'https://data.4dnucleome.org/'}

# let's search for all biosamples
# hits is a list of metadata dictionaries
# Let's work with experiment sets (the default). We should grab facet information
# first though. 'facet_info' keys will be the possible facets and each key contains
# the possible values with their counts
facet_info = ff_utils.get_item_facet_values('ExperimentSetReplicate', key=key)

# now specify kwargs - say we want to search for all experiments under the 4DN
# project that are of experiment type 'Dilution Hi-C'
kwargs = {
  'Project': '4DN',
  'Experiment Type': 'Dilution Hi-C'
}
results = ff_utils.faceted_search(**kwargs, key=key)

for key, value in results[0].items():
    print(key)
    print(f'\t{value}\n')

# # you can also search other types by specifying 'item_type' in kwargs
# # say we'd like to search for all users affiliated with the 4DN Testing Lab
# kwargs = {
#   'item_type' = 'user',
#   'Affiliation' = '4DN Testing Lab'
# }
# results = ffutils.faceted_search(**kwargs)

# # you can also perform negative searches by pre-pending '-' to your desired value
# # ie: get all users not affiliated with the 4DN Testing Lab
# # note that you can combine this with 'positive' searches as well
# kwargs = {
# 'item_type' = 'user',
# 'Affiliation' = '-4DN Testing Lab'
# }

# # You can also specify multiple pipe (|) seperated values for a field
# # ie: get all experiments sets from 4DN or External
# kwargs = {
#   'Project': '4DN|External'
# }
    