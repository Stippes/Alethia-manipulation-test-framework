from scripts.input_parser import parse_json_chat, parse_txt_chat, standardize_format
import logging
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

setup_logging()

raw1 = parse_json_chat("data/example.json")
std1 = standardize_format(raw1)
logger.info("Standardized format json %s", std1)

raw2 = parse_txt_chat("data/example.txt")
std2 = standardize_format(raw2)
logger.info("Standardized format txt %s", std2)
