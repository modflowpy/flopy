"""
Module for input/output utilities
"""
import numpy as np


def line_parse(line):
    """
    Convert a line of text into to a list of values.  This handles the
    case where a free formatted MODFLOW input file may have commas in
    it.

    """
    line = line.replace(',', ' ')
    return line.strip().split()


