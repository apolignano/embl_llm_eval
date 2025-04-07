import re
import logging

def regex_filter(label: str, pattern, all=True):
    try:
        return re.findall(pattern, label) if all else re.search(pattern, label).group()
    except Exception as e:
        logging.debug(e)

