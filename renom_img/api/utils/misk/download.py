import os
from tqdm import tqdm
import urllib.request


def download(url, save_path=None):
    """
    Download method with progress bar.

    Args:
        url (string): Download url.
        save_path (string): If not passed, the url's base name will be used and it'll be saved to current directory.
    """

    # TODO: Write error handling.
    filename = os.path.basename(url)
    if save_path is None:
        save_path = filename
    request = urllib.request.urlopen(url=url)
    filesize = int(request.headers['Content-length'])
    bar = tqdm(total=filesize, unit='B', unit_scale=True,
        unit_divisor=1024, desc="Download %s"%filename)
    def progress(block_count, block_size, total_size):
        percentage = block_count * block_size
        bar.update(percentage - bar.n)
    urllib.request.urlretrieve(url, filename=save_path, reporthook=progress)
    bar.n = filesize
    bar.close()


if __name__ == "__main__":
    download("https://www.python.org/ftp/python/3.5.5/Python-3.5.5.tgz")
