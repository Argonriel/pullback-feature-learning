"""Derive a consolidated cost ledger without changing any fitted procedure."""
import csv,json
import runner

def read(name):return list(csv.DictReader((runner.OUT/name).open(newline='')))
def main():
    runner.verify();runner.verify_commit('MODEL_COMMIT.json');runner.verify_commit('FIT_COMMIT.json');runner.verify_commit('SCORE_COMPLETE.json')
    training=read('training.csv');fits=read('temperature_fits.csv');actual=read('actual_training_work.csv');deploy=read('deployment_ledger.csv');out=[]
    for row in training:
        for pipeline in ('raw','full'):
            fit=next(r for r in fits if r['pair']==row['pair'] and r['procedure']==row['procedure'] and r['pipeline']==pipeline)
            derivatives=int(fit['iterations'])+3
            out.append(dict(pair=int(row['pair']),procedure=row['procedure'],pipeline=pipeline,unique_fitting_labels=int(row['unique_fit_labels']),
              accepted_training_labels=int(row['n_train']),validation_labels=int(row['n_validation']),calibration_labels=int(row['n_calibration']),
              training_updates=int(row['updates_executed']),selected_update=int(row['selected_step']),
              training_example_updates=int(row['training_example_updates']),representation_forward_examples=int(row['standalone_representation_forward_examples']),jacobian_forward_examples=int(row['jacobian_forward_examples']),
              jacobian_rows=int(row['jacobian_rows']),jacobian_scalar_entries=int(row['jacobian_scalar_entries']),loss_derivative_examples=int(row['loss_derivative_examples']),
              reverse_mode_backward_calls=0,lu_144_solves=int(row['lu_solves_144']),training_nominal_arithmetic_flops=int(row['nominal_arithmetic_flops']),
              training_tanh_evaluations=int(row['tanh_scalar_evaluations']),training_exp_evaluations=int(row['exp_scalar_evaluations']),training_logaddexp_evaluations=int(row['logaddexp_scalar_evaluations']),
              validation_forward_examples=int(row['validation_forward_examples']),standalone_calibration_forward_examples=int(fit['standalone_forward_examples']),
              scalar_temperature_parameters=1,temperature_derivative_evaluations=derivatives,temperature_objective_evaluations=2,
              temperature_derivative_label_evaluations=derivatives*int(fit['n_labels']),temperature_objective_label_evaluations=2*int(fit['n_labels']),
              deployment_forward_examples_per_item=1 if pipeline=='raw' else 2))
    runner.cw(runner.OUT/'standalone_cost_ledger.csv',out)
    samplings=[json.loads((runner.OUT/'splits'/f'{pair}_sampling.json').read_text()) for pair in range(1,17)]
    runner.jo(runner.OUT/'COST_TOTALS.json',dict(utc=runner.now(),
      actual_experimental_unique_fitting_labels=16*7424,accepted_training_label_union=16*2304,validation_label_union=16*1024,calibration_label_union=16*4096,
      synthetic_training_rejection_proposals=sum(m['proposals'] for r in samplings for m in r['metadata']),
      actual_training_example_updates=sum(int(r['training_example_updates']) for r in actual),actual_training_nominal_arithmetic_flops=sum(int(r['nominal_arithmetic_flops']) for r in actual),
      actual_lu_solves=sum(int(r['lu_solves_144']) for r in actual),actual_training_diagnostic_forward_examples=sum(int(r['training_diagnostic_forward_examples']) for r in actual),
      actual_validation_forward_examples=16*4096,actual_shared_calibration_forward_examples=sum(int(r['shared_actual_forward_examples']) for r in fits if r['pipeline']=='raw'),
      actual_temperature_derivative_evaluations=sum(int(r['iterations'])+3 for r in fits),actual_temperature_objective_evaluations=2*len(fits),
      actual_temperature_derivative_label_evaluations=sum((int(r['iterations'])+3)*int(r['n_labels']) for r in fits),actual_temperature_objective_label_evaluations=sum(2*int(r['n_labels']) for r in fits),
      actual_test_forward_examples=sum(int(r['inference_forward_examples']) for r in deploy),actual_test_invocations=sum(int(r['inference_invocations']) for r in deploy),
      notes='Procedures share trajectories/checkpoints within the experiment. Standalone ledgers count each alternative independently. Calibration raw/full outputs are shared within a procedure but physically recomputed across procedures. Preflight and isolated timing are extra audit work recorded separately. Scalar temperature derivatives are two bound evaluations, iterations bisection evaluations, and one final evaluation; there are two objective evaluations per fit.'))

if __name__=='__main__':main()
