"""Command line interface submodule."""


def gen_help(base, variables):
    varstr = ''.join('\n{}: {}'.format(key, val) for key, val in variables.items())
    return base+'\n\n\b'+varstr
