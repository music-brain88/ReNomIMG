import os
from tqdm import tqdm
import urllib.request
from renom_img.api.utility.exceptions.exceptions import *


def download(url, save_path=None):
    """
    Download method with progress bar.

    Args:
        url (string): Download url.
        save_path (string): If not passed, the url's base name will be used and it'll be saved to current directory.
    """

    # TODO: Write error handling.
    if url is None:
        raise WeightNotFoundError('Weight can not be downloaded, URL is None.')
    filename = os.path.basename(url)
    if save_path is None:
        save_path = filename
    try:
        request = urllib.request.urlopen(url=url)
        filesize = int(request.headers['Content-length'])
    except:
        raise WeightURLOpenError(
            'Weight URL can not be opened. Check if the file exists in {}'.format(url))
    bar = tqdm(total=filesize, unit='B', unit_scale=True,
               unit_divisor=1024, desc="Download %s" % filename)

    def progress(block_count, block_size, total_size):
        percentage = block_count * block_size
        bar.update(percentage - bar.n)
    try:
        urllib.request.urlretrieve(url, filename=save_path, reporthook=progress)
    except:
        raise WeightRetrieveError(
            'Weight file can not be retrieved. Check if the url is valid or not.{}'.format(url))
    bar.n = filesize
    bar.close()
