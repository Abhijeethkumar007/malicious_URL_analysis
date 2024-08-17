import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

def extract_features(url):
    count_dots = url.count('.')
    count_slashes = url.count('/')
    count_special_chars = len(re.findall(r'[^a-zA-Z0-9]', url)) - count_dots - count_slashes
    url_length = len(url)
    num_digits = len(re.findall(r'\d', url))
    num_links = len(re.findall(r'https?://', url))
    scheme_ftp = int(urlparse(url).scheme == 'ftp')
    scheme_http = int(urlparse(url).scheme == 'http')
    scheme_https = int(urlparse(url).scheme == 'https')
    
    return [count_dots, count_slashes, count_special_chars, url_length, num_digits, num_links, scheme_ftp, scheme_http, scheme_https]



def get_prediction_from_url_rf(test_url):
    features_test = extract_features(test_url)
    features_test_df = pd.DataFrame([features_test], columns=[
        'count_dots', 'count_slashes', 'count_special_chars', 'url_length',
        'num_digits', 'num_links', 'scheme_ftp', 'scheme_http', 'scheme_https'
    ])
    
    return features_test_df
