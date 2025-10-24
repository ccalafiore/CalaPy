import datetime
import importlib

__version__ = '0.2.0.3'


# TODO: Get the release timestamp automatically from PyPI???
__release_day__ = 24
__release_month_num__ = 10
__release_year__ = 2025


__release_date_object__ = datetime.date(__release_year__, __release_month_num__, __release_day__)
__release_date__ = __release_date_object__.__format__('%d %B %Y')
__release_month_name__ = __release_date_object__.__format__('%B')
del datetime

__author__ = 'Calafiore Carmelo'
__author_email__ = 'dr.carmelo.calafiore@gmail.com'
__maintainer_email__ = 'dr.carmelo.calafiore@gmail.com'

submodules = [
    'array', 'check', 'clock', 'combinations', 'directory', 'download',
    'format', 'image', 'lists', 'maths', 'mixamo', 'ml', 'pkl', 'plot', 'preprocess',
    'shutdown', 'stats', 'stimulation', 'strings', 'threading', 'txt']
others = []
__all__ = submodules + others

for sub_module_m in submodules:
    importlib.import_module(name='.' + sub_module_m, package=__package__)
