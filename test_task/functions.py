import numpy as np
import re
import logging

def regex_filter(label: str, pattern, all=True):
    try:
        return re.findall(pattern, label) if all else re.search(pattern, label).group()
    except Exception as e:
        logging.debug(e)

def compute_set_precision(results, targets):
    eval_targets = eval(targets)
    tp_list = list(set(eval_targets).intersection(set(results)))
    fp_list = list(set(results).difference(set(eval_targets)))
    tp = len(tp_list)
    fp = len(fp_list)
    if (tp + fp > 0):
        return tp/(tp + fp)
    else:
        return 0

def compute_set_recall(results, targets):
    eval_targets = eval(targets)
    tp_list = list(set(eval_targets).intersection(set(results)))
    fn_list = list(set(eval_targets).difference(set(results)))
    tp = len(tp_list)
    fn = len(fn_list)
    if (tp + fn > 0):
        return tp/(tp + fn)
    else:
        return 0