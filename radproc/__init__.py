"""radar data processing tools"""
from radproc._version import __version__

NAN_REPLACEMENT = {'ZH': -10, 'ZDR': 0, 'KDP': 0, 'RHO': 0, 'DP': 0,
                   'PHIDP': 0, 'MLI': 0, 'T':-999}
