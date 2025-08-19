
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

def load_hmm_export(json_path: str | Path) -> Dict[str, Any]:
    """
    Load HMM export (including transition matrices) created like:

    params = {
        'ret': [ret],
        '2s': {
            'params': [means, covars],
            't_mat': [A, pi],
            'hidden': hidden_states,
            'probs': probs
        },
        '3s': {
            'params': [means, covars],
            't_mat': [A_3, pi_3],
            'hidden': hidden_states_3,
            'probs': probs_3
        },
        'Dates': ['YYYY-MM-DD', ...]
    }

    Returns:
        {
            'ret': returns
          'dates': DatetimeIndex,
          '2s': {
            'means': np.ndarray,
            'covars': np.ndarray,
            'A': np.ndarray,        # transition matrix
            'pi': np.ndarray,       # start probabilities
            'hidden': np.ndarray,
            'probs': np.ndarray
          },
          '3s': { ... same keys ... }
        }
    """
    json_path = Path(json_path)
    with json_path.open('r') as f:
        data = json.load(f)

    def _block(d):
        means, covars = d['params']
        A, pi = d['t_mat']
        return {
            'means':  np.asarray(means),
            'covars': np.asarray(covars),
            'A':      np.asarray(A),
            'pi':     np.asarray(pi),
            'hidden': np.asarray(d['hidden']),
            'probs':  np.asarray(d['probs'])
        }

    out = {
        'ret': np.asarray(data['ret']),
        'dates': pd.to_datetime(pd.Index(data['Dates'])),
        '2s': _block(data['2s']),
        '3s': _block(data['3s'])
    }
    return out


# load
hmm_data = load_hmm_export('/Users/stefanochiapparini/Desktop/PYTHON/Finance/API/HMM/states_sp500.json')

dates = hmm_data['dates']
A2, pi2 = hmm_data['2s']['A'], hmm_data['2s']['pi']
A3, pi3 = hmm_data['3s']['A'], hmm_data['3s']['pi']
probs2 = hmm_data['2s']['probs']          # shape: (n_obs, 2)
hidden3 = hmm_data['3s']['hidden']        # shape: (n_obs,)
ret = hmm_data['ret']

means_2 = hmm_data['2s']['means']
cov_2 = hmm_data['2s']['covars']


means_3 = hmm_data['3s']['means']
cov_3 = hmm_data['3s']['covars']

#print(means_2)
#print(means_3)
#print(cov_2)
#print(cov_3)
#print(A2)
#print(A3)

probs2 = hmm_data['2s']['probs'] 

