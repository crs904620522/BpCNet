import os
import urllib
import torch
from torch.utils import model_zoo


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''

    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filename):
        '''Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print('=> Loading checkpoint from local file...',filename)
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url):
        '''Load a module dictionary from url.

        Args:
            url (str): url to saved model
        '''
        print('=> Loading checkpoint from url...',url)
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    '''
        try:
            for k, v in self.module_dict.items():
                if k in state_dict:
                    v.load_state_dict(state_dict[k])
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)
            scalars = {k: v for k, v in state_dict.items()
                       if k not in self.module_dict}
        except Exception as e:
            for k, v in self.module_dict.items():
                if k in state_dict:
                    # v.load_state_dict(state_dict[k])
                    try:
                        if k == 'model':
                            model_dict = v.state_dict()
                            # print(model_dict)
                            weight_dict = state_dict[k]
                            load_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
                            model_dict.update(load_dict)
                            # model.load_state_dict(model_dict, strict=False)
                            v.load_state_dict(model_dict)
                        else:
                            v.load_state_dict(state_dict[k])
                    except Exception as e:
                        print(k, "error", e)
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)

            scalars = {k: v for k, v in state_dict.items()
                       if k not in self.module_dict}
        return scalars


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
