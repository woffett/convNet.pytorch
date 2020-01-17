import getpass
import os
import pathlib
import socket

def is_windows():
    """Determine if running on windows OS."""
    return os.name == 'nt'

def get_base_dir():
    hostname = socket.gethostname()
    if is_windows():
        username = ('Avner' if (hostname == 'Avner-X1Carbon')
                    else 'avnermay')
        path = 'C:\\Users\\{}\\Babel_Files\\distill'.format(username)
    # elif hostname == 'DN0a22a222.SUNet':
    # elif 'DN' in hostname and '.SUNet' in hostname:
    elif '.stanford.edu' in hostname:
        path = '/dfs/scratch0/avnermay/distill'
    else:
        path = '/proj/distill'
    return path