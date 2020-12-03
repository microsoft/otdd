import os
from os.path import dirname, abspath
import logging
ROOT_DIR = dirname(dirname(abspath(__file__))) # Project Root
HOME_DIR = os.getenv("HOME") # User home dir
from .utils import launch_logger
