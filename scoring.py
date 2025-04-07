import numpy as np

def std(label):
    return str(label).strip().lower()

def is_match(inputs, targets):
    for target, result in zip(inputs, targets): 
        if (std(target) == std(result)): # Add or condition when thresholding is applied
            return 1
        return 0 

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
    
def compute_set_f1(recall: float, precision: float):
    denominator = recall + precision
    if (denominator > 0):
        return 2*(recall*precision)/denominator
    else:
        return 0

def mean(scores: list[float]):
    return np.mean(scores)

def median(scores: list[float]):
    return np.median(scores)