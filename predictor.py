import torch
import numpy as np
import utils
import argparse
from parse_config import ConfigParser
import model.model as module_arch

class DigitPredictor:

    def __init__(self):
        self._loadConfig()
        self._loading_model() 
    def predict_on_image(self, image_or_fileurl):
        if isinstance(image_or_fileurl, str):
            image = utils.read_image(image_or_fileurl,grayscale=True)
        else:
            image = image_or_fileurl
        
        return torch.argmax(self.model(image), dim=1).item()
        

    def _loading_model(self):
        logger = self.config.get_logger('test')
        model = self.config.init_obj('arch', module_arch)
        logger.info(model)


        logger.info('Loading checkpoint: {} ...'.format(self.config.resume))
        checkpoint = torch.load(self.config.resume)
        state_dict = checkpoint['state_dict']
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
    
    def _loadConfig(self):
        args = argparse.ArgumentParser(description='Dummy Args')
        args.add_argument('-r', '--resume', default="weights/deploy_model.pth", type=str,help='path to deployable checkpoint (default: None)')

        self.config = ConfigParser.from_args(args, deploy=True)




            


