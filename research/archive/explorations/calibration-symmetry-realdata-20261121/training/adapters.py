"""Portable calibrated-model interface for selectors and robustness checks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('training_adapter_model',HERE/'model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
task=model.task
FAMILIES=('legacy_pb','selected_pb','selected_gd')
VARIANTS=('legacy_raw','legacy_temp','pb_raw','pb_temp','gd_raw','gd_temp')
FAMILY_FOR_VARIANT={name:FAMILIES[i//2] for i,name in enumerate(VARIANTS)}

def get_adapter(beta=1.):
    beta=float(beta)
    if not np.isfinite(beta) or beta<=0:raise ValueError('Positive finite inverse temperature required')
    def constants(theta):
        L,C=task.network_constants(theta)
        # Conservative residual bound relative to the UNscaled x1 on this compact support.
        return beta*L,beta*C+abs(beta-1)*1.4
    def difference_bound(theta):
        L,C=task.network_constants(theta);return beta*min(1.4+C,1.4*L)
    return SimpleNamespace(name='two_invariants',dimension=4,parameters=144,quotient_dim=2,beta=beta,
      network=task.core,predict=lambda theta,x:beta*task.predict(theta,x),partner=task.partner,transform=task.transform,
      quotient=task.quotient,angle=task.angle,coordinates=task.coordinates,sample_inputs=task.sample_inputs,
      posterior=task.posterior,network_constants=constants,base_network_constants=task.network_constants,
      difference_bound=difference_bound,logit_bound=lambda theta:beta*(1.4+task.network_constants(theta)[1]),
      support_lo=np.array([.4,.4]),support_hi=np.array([1.4,1.4]))

def load_variant(pair,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    family=FAMILY_FOR_VARIANT[variant]
    state_path=HERE/'states'/f'{pair}.npz'
    theta=np.load(state_path)[family]
    temperature_path=HERE/'temperatures'/f'{pair}.json'
    beta=json.loads(temperature_path.read_text())['families'][family]['beta'] if variant.endswith('_temp') else 1.
    return get_adapter(beta),theta,dict(pair=pair,variant=variant,family=family,beta=beta,temperature=1/beta,
      state_path=str(state_path),temperature_path=str(temperature_path))

def policy_paths(pair,variant):
    return {name:HERE/'policies'/f'{pair}_{variant}_{name}.npz' for name in ('grid','folded')}
