import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from urllib.request import urlopen, urlretrieve 
from pathlib import Path
import cv2
import os 
import numpy as np
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def read_image(image_uri, grayscale=False):                      
    """Read image from url or file path and return image array                                 
      Customize as require"""                                                                  
    def read_image_from_filename(image_filename, imread_flag):                                 
        return cv2.imread(str(image_filename), imread_flag)                                    
                                                                                               
    def read_image_from_url(image_url, imread_flag):                                           
        url_response = urlopen(str(image_url))  # nosec                                        
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)                   
        return cv2.imdecode(img_array, imread_flag)                                            
                                                                                               
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR                      
    local_file = os.path.exists(image_uri)                                                     
    try:                                                                                       
        img = None                                                                             
        if local_file:                                                                         
            img = read_image_from_filename(image_uri, imread_flag)                             
        else:                                                                                  
            img = read_image_from_url(image_uri, imread_flag)    

        img = cv2.resize(img, (28, 28)) 
        img = torch.from_numpy(img)
        img = img.view(1,28,28)
        img = img.unsqueeze(0).to("cuda").float()                 
        assert img is not None                                                                 
    except Exception as e:                                                                     
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))                
    return img  


                                                                                               
def read_b64_image(b64_string, grayscale=False):                                               
    """Load base64-encoded images"""                                                           
    import base64                                                                              
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR                      
    try:                                                                                       
        _, b64_data = b64_string.split(',')                                                    
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)  
    except Exception as e:                                                                     
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, e))         

def write_image(image, filename):
    cv2.imwrite(str(filename), image)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
