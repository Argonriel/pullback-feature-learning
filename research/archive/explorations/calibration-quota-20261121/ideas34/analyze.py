"""Summaries, algebraic audits, and scientific figures from frozen ideas3/4 data."""
import csv,json,itertools
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import study as s

ROOT=Path(__file__).resolve().parent
def read(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(a,rng):
    a=np.array(a);idx=rng.integers(0,len(a),(10000,len(a)));return np.quantile(a[idx].mean(1),[.025,.975])

def numerical_audit():
    rng=np.random.default_rng(7736000);errors=[]
    for trial in range(100):
        b=rng.normal(size=5);w=rng.dirichlet(np.ones(5));pi=float(rng.uniform(.01,.99));C=float(rng.normal())
        lo,hi,_,_=s.risk_interval(C,b,w,pi);vertices=[]
        for free in range(5):
            fixed=[j for j in range(5) if j!=free]
            for values in itertools.product((0.,1.),repeat=4):
                p=np.zeros(5);p[fixed]=values;p[free]=(pi-p@w)/w[free]
                if -1e-12<=p[free]<=1+1e-12:vertices.append(C-w@(p*b))
        errors.append(max(abs(lo-min(vertices)),abs(hi-max(vertices))))
    duals=[];direct=[]
    for path in (ROOT/'calibration').glob('*_finite.npz'):
        d=np.load(path);sub=d['subsets'];h=d['h'];m=d['m']
        for r in range(16):
            b=sub@(h[r]-m[r])/16;c=sub@(s.soft(h[r])-s.soft(m[r]))/16
            left=d['left'][r];right=d['right'][r];mix=d['mix'][r]
            bl=left@(h[r]-m[r])/16;br=right@(h[r]-m[r])/16
            cl=left@(s.soft(h[r])-s.soft(m[r]))/16;cr=right@(s.soft(h[r])-s.soft(m[r]))/16
            intercept=mix*cl+(1-mix)*cr
            if abs(br-bl)>1e-14:lam=(cr-cl)/(br-bl)
            else:
                lower=max((c[b>1e-14]-intercept)/b[b>1e-14],default=-1e6)
                upper=min((c[b< -1e-14]-intercept)/b[b< -1e-14],default=1e6)
                lam=(lower+upper)/2
            duals.append(float(np.max(c-lam*b-intercept)))
            direct.append(abs(mix*bl+(1-mix)*br))
    assert max(errors)<1e-12 and max(duals)<1e-12 and max(direct)<1e-12
    return dict(known_prevalence_unequal_weight_vertex_checks=100,max_fractional_allocation_bound_error=max(errors),
        finite_conditional_LP_dual_certificates=len(duals),max_dual_optimality_violation=max(duals),max_expected_conditional_imbalance=max(direct))

def region_diagnostics():
    # Derived summaries of the already-frozen held-out inputs, not new selection.
    rows=[];first=[];models=np.load(s.MODELS)
    for i,key in enumerate(s.CONFIG['old_model_keys'],1):
        x=s.sample_inputs(s.CONFIG['test_seed_base']+i,s.CONFIG['n_test']);r=np.linalg.norm(x,axis=1)
        canonical=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0)))[:,None],-x,x)
        phi=np.arctan2(canonical[:,1],canonical[:,0]);ri=np.minimum((r/3*16).astype(int),15);ai=np.minimum((phi/np.pi*16).astype(int),15);ids=ri*16+ai
        h=s.core.representation(models[key],x)[:,0];b=s.core.representation(models[key],-x)[:,0];J=s.gap(h,b)
        found=False
        for ep in s.CONFIG['epsilons']:
            gain=J-ep*(x[:,0]/r)*(h-b)/4
            local=[]
            for cell in range(256):
                v=gain[ids==cell]
                if len(v)<32:continue
                rec=dict(model=key,epsilon=ep,radius_lo=(cell//16)*3/16,radius_hi=(cell//16+1)*3/16,
                    folded_angle_lo=(cell%16)*np.pi/16,folded_angle_hi=(cell%16+1)*np.pi/16,n=len(v),mean_gain=float(v.mean()),se=float(v.std(ddof=1)/np.sqrt(len(v))))
                rows.append(rec);local.append(rec)
            harmful=[z for z in local if z['mean_gain']< -1.96*z['se']-1e-12]
            if harmful and not found:
                best=min(harmful,key=lambda z:z['mean_gain']);first.append(best);found=True
    write(ROOT/'results/idea3_all_populated_regions.csv',rows);write(ROOT/'results/idea3_first_tested_harmful_region.csv',first)
    return first

def main():
    results=ROOT/'results';r3=read(results/'idea3_by_model.csv');r4=read(results/'idea4_intervals.csv');rng=np.random.default_rng(s.CONFIG['bootstrap_seed'])
    summary3=[]
    for ep in s.CONFIG['epsilons']:
        for policy in ('raw','always','random64','safe64','forced64','oracle_safe64'):
            rr=[r for r in r3 if float(r['epsilon'])==ep and r['policy']==policy];a=np.array([float(r['gain']) for r in rr]);bounds=ci(a,rng)
            summary3.append(dict(epsilon=ep,policy=policy,n_old_models=len(rr),mean_gain=float(a.mean()),gain_ci_lo=float(bounds[0]),gain_ci_hi=float(bounds[1]),
                mean_risk=float(np.mean([float(r['risk']) for r in rr])),mean_selected_fraction=float(np.mean([float(r['selected_fraction']) for r in rr])),
                mean_calls=float(np.mean([float(r['model_evaluations_per_input']) for r in rr])),models_with_net_harm=int((a< -1e-12).sum()),
                mean_harmful_selected_fraction=float(np.mean([float(r['harmful_selected_fraction']) for r in rr])),
                positive_despite_negative_point_bound=float(np.mean([float(r['positive_gain_despite_negative_point_bound']) for r in rr])),
                positive_despite_negative_cell_bound=float(np.mean([float(r['positive_gain_despite_negative_cell_bound']) for r in rr]))))
    write(results/'idea3_summary.csv',summary3)
    summary4=[]
    for name in ('radius_only','antipodal_pairs','confidence','balanced_orientation'):
        for constraint in ('unrestricted','prevalence_half'):
            rr=[r for r in r4 if r['policy']==name and r['constraint']==constraint]
            summary4.append(dict(policy=name,constraint=constraint,n_old_models=len(rr),mean_C=float(np.mean([float(r['c']) for r in rr])),
                mean_lower=float(np.mean([float(r['lower']) for r in rr])),mean_upper=float(np.mean([float(r['upper']) for r in rr])),
                mean_width=float(np.mean([float(r['width']) for r in rr])),models_with_possible_both_signs=sum(float(r['lower'])< -1e-12 and float(r['upper'])>1e-12 for r in rr),
                maximum_conditional_imbalance=max(float(r['max_abs_conditional_b']) for r in rr)))
    write(results/'idea4_summary.csv',summary4)
    aud=json.loads((results/'idea4_deployment_audit.json').read_text());adv=np.array([r['balanced_minus_antipodal_gain'] for r in aud]);adci=ci(adv,rng)
    ronly={r['model']:float(r['c']) for r in r4 if r['policy']=='radius_only' and r['constraint']=='unrestricted'}
    balanced={r['model']:float(r['c']) for r in r4 if r['policy']=='balanced_orientation' and r['constraint']=='unrestricted'}
    diff=np.array([balanced[k]-ronly[k] for k in balanced]);diffci=ci(diff,rng)
    comparisons=dict(balanced_minus_antipodal_mean=float(adv.mean()),ci95=adci.tolist(),wins=int((adv>1e-12).sum()),
        balanced_minus_radius_only_mean=float(diff.mean()),radius_comparison_ci95=diffci.tolist(),wins_over_radius_only=int((diff>1e-12).sum()),
        largest_expected_abs_b=max(r['max_expected_balance'] for r in aud),minimum_marginal_asymmetry=min(r['marginal_asymmetry'] for r in aud))
    s.dump(results/'idea4_comparisons.json',comparisons);s.dump(results/'independent_algebraic_audit.json',numerical_audit());first=region_diagnostics()
    figures(summary3,summary4)
    report(summary3,summary4,comparisons,first)
    print(json.dumps(comparisons,indent=2))

def figures(s3,s4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,ax=plt.subplots(1,3,figsize=(13.2,3.5),gridspec_kw={'width_ratios':[1.1,1,1.05]})
    labels={'always':'Always half-turn (2 calls)','safe64':'Certified optional quota','forced64':'Forced quota64','oracle_safe64':'Pointwise oracle (2-call diagnostic)'}
    colors={'always':'#546d8d','safe64':'#16866d','forced64':'#bf4b45','oracle_safe64':'#999999'}
    for p in labels:
        rr=[r for r in s3 if r['policy']==p];ep=[r['epsilon'] for r in rr];v=[r['mean_gain'] for r in rr]
        ax[0].plot(ep,v,'o-',color=colors[p],label=labels[p],ms=3,lw=1.7)
        if p!='oracle_safe64':ax[1].plot(ep,[100*r['mean_selected_fraction'] for r in rr],'o-',color=colors[p],ms=3)
    ax[0].axhline(0,color='black',lw=.6);ax[0].set(title='Approximate symmetry: actual gain',xlabel='Maximum antipodal mismatch epsilon',ylabel='Gain in log loss (nats/input)');ax[0].legend(fontsize=7,loc='lower left')
    ax[1].set(title='Certificate coverage and forced use',xlabel='Maximum antipodal mismatch epsilon',ylabel='Inputs receiving second view (%)',ylim=(-2,103))
    rr=[r for r in s4 if r['constraint']=='prevalence_half'];names=['Radius only','Antipodal pairs','Confidence','Balanced orientation']
    for j,r in enumerate(rr):
        ax[2].plot([r['mean_lower'],r['mean_upper']],[j,j],color='#bf4b45' if r['policy']=='confidence' else '#16866d',lw=5,solid_capstyle='round')
        ax[2].scatter([(r['mean_lower']+r['mean_upper'])/2],[j],s=28,color='#bf4b45' if r['policy']=='confidence' else '#16866d')
    ax[2].axvline(0,color='black',lw=.6);ax[2].set_yticks(range(4),names);ax[2].invert_yaxis();ax[2].set(title='Mean per-model sharp ranges\nKnown prevalence 1/2',xlabel='Gain allowed by unknown radial labels')
    fig.suptitle('Ideas3–4:16 fixed historical models; finite-grid policy ranges are a separate population',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'ideas34_results.png',bbox_inches='tight');fig.savefig(ROOT/'ideas34_results.pdf',bbox_inches='tight');plt.close(fig)

def report(s3,s4,comparisons,first):
    index={(r['epsilon'],r['policy']):r for r in s3};cost=read(ROOT/'calibration/costs_and_balance.csv');math=json.loads((ROOT/'results/mathematical_checks.json').read_text());audit=json.loads((ROOT/'results/independent_algebraic_audit.json').read_text())
    lines=['# Approximate symmetry and identifiable selective gains','',
    'These checks use sixteen fixed, previously inspected historical PB500 models. They are separate from the fresh-model2A/2B experiments. We froze the source, epsilon grid, policies and model hashes before held-out scoring. Theoretical identification and numerical sampling precision are reported separately.','',
    '## Idea3: a valid certificate can become conservative','',
    'For an antipodal pair, exact gain is J−(p−q)(a−b)/4. Under |p−q|≤epsilon, J−epsilon|a−b|/4 is sharp. Our executable gate never obtains the second logit before deciding. Offline interval propagation supplies a conservative cellwise certificate, indexed by radius and folded angle. It selects at most64 of256 only where the bound is strictly positive. Zero certificates are deliberately skipped; at epsilon0 the symmetry theorem would also permit filling remaining capacity without harm.','',
    '|epsilon|Always-half-turn gain|Certified gain|Certified usage|Forced64 gain|Forced models harmed|',
    '|---:|---:|---:|---:|---:|---:|']
    for ep in s.CONFIG['epsilons']:
        a=index[ep,'always'];b=index[ep,'safe64'];c=index[ep,'forced64']
        lines.append(f"|{ep:g}|{a['mean_gain']:.7f}|{b['mean_gain']:.7f}|{100*b['mean_selected_fraction']:.3f}%|{c['mean_gain']:.7f}|{c['models_with_net_harm']}/16|")
    lines+=['',
    'Every selected pair of the certified rule had nonnegative measured gain, consistent with the pointwise guarantee. Its useful coverage collapses as the allowed label mismatch grows: this is a real limitation, not hidden by calling abstention a practical success. Forcing64 calls can include uncertified harmful pairs. At epsilon0.1, five models had negative forced-policy mean gain even though universal averaging was beneficial on every model; the forced rule is an illustrative lower-bound ranking, not claimed to be an optimal shift-aware policy.','',
    'A negative pointwise or cell bound does not imply actual harm. The archived rows report both positive-gain/negative-bound frequencies. The full second-view policy first had negative cohort mean at tested epsilon0.5. Coarse-region diagnostics preserve every region with at least32 held-out inputs; their first tested harmful regions are exploratory descriptive localizations, not independently confirmed thresholds. Known p_epsilon is used only to evaluate risk, never by the deployment gate.','',
    f"Offline bounds use16,384 cells and32,768 interval propagations per model; median construction time was{np.median([float(r['interval_seconds']) for r in cost]):.4f}s. Interval operations are not ordinary model forward evaluations. Executed gates call the model once on256 inputs then only on their selected subset. The pointwise-bound oracle already needs both logits on every input and is explicitly charged2calls/input. All policies share independent held-out inputs within a model.",'',
    '## Idea4: orientation sensitivity need not destroy identification','',
    'For a fixed rule, gain is C−E[p(R)b(R)]. With unrestricted unknown radial p, its exact sharp range is [C−E b_+,C+E b_-]; exact identification holds iff b=0 almost surely. With known prevalence pi strictly between0and1, identification holds iff b is constant; sharp bounds allocate positive-label probability to the top or bottom pi-mass of b. The mathematical proof and fractional-atom treatment are in THEORY.md.','',
    'The finite construction has16radii×16orientations per stratified batch. A mixture of atmost2 four-point subsets at each radius gives exactly64 extra calls in every realization and conditional signed balance in expectation. Its marginal selection probabilities truly differ at antipodes. The table covers all256support points, so this is an exact finite-support construction with explicit calibration and caching qualifications, not a continuous deployment claim.','',
    '|Policy|Mean gain range with prevalence1/2|Models admitting both signs|',
    '|---|---:|---:|']
    for r in s4:
        if r['constraint']=='prevalence_half':lines.append(f"|{r['policy']}|[{r['mean_lower']:.6f},{r['mean_upper']:.6f}]|{r['models_with_possible_both_signs']}/16|")
    lines+=['',
    f"The balanced orientation-sensitive rule improved over the equally constrained antipodal-pair rule by{comparisons['balanced_minus_antipodal_mean']:.7f}nats/input on average (descriptive paired95%interval{comparisons['ci95']}, wins{comparisons['wins']}/16). Its conditional imbalance was at most{comparisons['largest_expected_abs_b']:.2e}; every model had marginal antipodal selection asymmetry at least{comparisons['minimum_marginal_asymmetry']:.3f}.",'',
    f"It did not dominate allocation across radii: its mean difference versus the radius-only rule was{comparisons['balanced_minus_radius_only_mean']:.7f}nats/input, winning{comparisons['wins_over_radius_only']}/16. The radius-only rule can spend all16calls on a chosen radius, whereas the constructive balanced rule reserves4perradius. This makes the latter a conditional-balance feasibility/optimization result under its stated per-radius restriction, not the globally optimal64-call policy.",'',
    'Confidence admitted opposite gain signs for every model even when class prevalence was known to be1/2. Hidden evaluator posteriors include each policy’s actual extremizers; balanced and symmetric rules keep their exact gains across these worlds. Displayed mean interval endpoints average model-specific sharp bounds; their extremizing posteriors can differ across models, so no common adversarial posterior is claimed to attain all endpoints simultaneously. The finite policy used256offline support forwards per model. A complete support cache could eliminate subsequent model calls; our executed256+64 path deliberately reevaluates predictions and verifies the budget, without claiming a fundamental compute saving on this artificial support.','',
    '## Checks, counterexamples and remaining limits','',
    f"The four-point same-prevalence gains were{math['four_point_gains'][0]:.9f} and{math['four_point_gains'][1]:.9f}. Sharp epsilon-bound equality was checked to{math['max_sharp_epsilon_identity_error']:.2e}. One hundred unequal-weight fixed-prevalence problems matched exhaustive linear-program vertices to{audit['max_fractional_allocation_bound_error']:.2e}. All256conditional subset-mixture optima had valid numerical dual certificates, maximum violation{audit['max_dual_optimality_violation']:.2e}.",'',
    f"A single continuous-radius bin with average imbalance0 still admits same-prevalence gains{math['opposite_half_interval_gains'][0]:.6f} and{math['opposite_half_interval_gains'][1]:.6f}; smooth posterior alternatives give{math['smooth_opposite_gains'][0]:.6f} and{math['smooth_opposite_gains'][1]:.6f}. Finite-bin balance cannot stand in for conditioning on every radius. If a credible L-Lipschitz posterior bound is available, the residual ambiguity is at most L times bin width times E|b|/2. No such regularity is inferred from unlabeled scores.",'',
    'Randomized-batch repetitions estimate a fixed-model policy expectation; they are not extra trained-model replications. Paired descriptive intervals resample the16oldmodels. The graphical comparison combines two deliberately different input populations, clearly labeled; their absolute gains are not a cross-experiment ranking. No numerical failure or losing epsilon/policy was dropped.','',
    '![Scientific results](ideas34_results.png)','',
    '## Reproduction','',
    'Use NumPy, threadpoolctl and Matplotlib in the project environment. Keep the bundled source/ and legacy/ siblings. To reproduce without overwriting this archive, copy the source files into a fresh ideas34 directory with the same parent layout, then run:','',
    '```powershell','python study.py freeze','python study.py fit','python study.py evaluate','python analyze.py','```','',
    'The freeze, calibration manifest and scoring timestamps establish the recorded sequence. Raw per-model, per-batch, hidden-world, region, deployment and algebraic-audit records are under results/. The scientific figure is available as PNG and PDF.']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(1):main()
