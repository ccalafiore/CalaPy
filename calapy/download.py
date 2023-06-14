from os.path import isfile as os_path_isfile
from time import sleep
from . import clock as cc_clock


def wait_downloading(directory_saved_as, max_seconds_wait=60):

    timer = cc_clock.Timer()
    wait = True
    while wait:

        if os_path_isfile(directory_saved_as):
            wait = False
        elif timer.get_seconds() > max_seconds_wait:
            raise TimeoutError
        sleep(1)



