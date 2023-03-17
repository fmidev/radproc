"""quantitative precipitation estimation"""


def rainrate(z):
    """basic r(z) relation"""
    return 0.0292*z**(0.6536)
