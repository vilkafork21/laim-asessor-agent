import copy,hashlib,itertools,json,sys,warnings
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze_expansion import fast_metrics,measure,crossfit,checked_scores,PRIMARY,PROFILES,paired_bootstrap
from metrics_local import agreement
from strict_binding import bind
from trace_binding import validated_evidence
from evidence_packets import packet,verify_quote
from record_judgments import record
from reveal_gold import reveal
from calibration_sensitivity import monotone

@pytest.mark.parametrize('seed',range(30))
def test_fast_metrics_matches_general(seed):
    rng=np.random.default_rng(seed);y=rng.integers(0,3,96);p=rng.integers(0,3,96)
    general=agreement(y,p);fast=fast_metrics(y,p)
    for k in PRIMARY:assert fast[k]==pytest.approx(general[k],abs=1e-12)

@pytest.mark.parametrize('y,p',[
    ([2,2,2],[2,2,2]),([0,1,2],[2,2,2]),([2,2,2],[0,1,2]),([0,0,1,1],[1,1,2,2]),
    ([0,1,2],[0,1,2]),([0,1,2],[2,1,0])])
def test_metric_edge_cases(y,p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');g=agreement(y,p)
    f=fast_metrics(y,p)
    for k in PRIMARY:
        if g[k] is None:assert f[k] is None
        else:assert f[k]==pytest.approx(g[k],abs=1e-12)

@pytest.mark.parametrize('bad',[[0,.5,2],[0,3,2],[0,float('nan'),2],['0','1'],[True,False],[[0,1]],[]])
def test_invalid_scale(bad):
    if bad==[]:assert len(checked_scores(bad))==0
    else:
        with pytest.raises(ValueError):checked_scores(bad)

def trace():
    return {'messages':[{'type':'human','content':'question'},
        {'type':'ai','content':'','tool_calls':[{'id':'c1','name':'search_case_docs'}]},
        {'type':'tool','name':'search_case_docs','tool_call_id':'c1','status':'success','content':'source\nwith formatting'},
        {'type':'ai','content':'answer'}]}

def test_original_unchanged():
    t=trace();assert bind(t,'question','answer')==validated_evidence(t,'question','answer')

def test_openai_calls_supported():
    t=trace();t['messages'][1]['tool_calls']=[{'id':'c1','type':'function','function':{'name':'search_case_docs','arguments':'{}'}}]
    assert len(bind(t,'question','answer'))==1

@pytest.mark.parametrize('mutation',['duplicate_call','nonassistant_call','duplicate_result','missing_call_name','missing_call_id','bad_function'])
def test_malformed_trace_rejected(mutation):
    t=trace()
    if mutation=='duplicate_call':t['messages'][1]['tool_calls']*=2
    elif mutation=='nonassistant_call':t['messages'][1]['type']='tool'
    elif mutation=='duplicate_result':t['messages'].insert(3,copy.deepcopy(t['messages'][2]))
    elif mutation=='missing_call_name':del t['messages'][1]['tool_calls'][0]['name']
    elif mutation=='missing_call_id':del t['messages'][1]['tool_calls'][0]['id']
    else:t['messages'][1]['tool_calls'][0]['function']='invalid'
    with pytest.raises(ValueError):bind(t,'question','answer')

@pytest.mark.parametrize('mutation',['error','wrong_name','unknown_id','before_call','delegate'])
def test_untrusted_result_excluded(mutation):
    t=trace()
    if mutation=='error':t['messages'][2]['status']='error'
    elif mutation=='wrong_name':t['messages'][2]['name']='wrong'
    elif mutation=='unknown_id':t['messages'][2]['tool_call_id']='unknown'
    elif mutation=='before_call':t['messages'][1],t['messages'][2]=t['messages'][2],t['messages'][1]
    else:t['messages'][1]['tool_calls'][0]['name']=t['messages'][2]['name']='delegate_to_worker'
    assert bind(t,'question','answer')==[]

def test_future_ignored():
    t=trace();base=bind(t,'question','answer');t['messages'].append(copy.deepcopy(t['messages'][2]));assert bind(t,'question','answer')==base

@pytest.mark.parametrize('q,a',[('question ','answer'),('question','answer '),('other','answer'),('question','other')])
def test_exact_identity(q,a):
    with pytest.raises(ValueError):bind(trace(),q,a)

def test_error_status_was_accepted_by_legacy_research_adapter():
    t=trace();t['messages'][2]['status']='error';assert len(validated_evidence(t,'question','answer'))==1;assert bind(t,'question','answer')==[]

def test_empty_tool_return_is_not_evidence_text():
    e=bind(trace(),'question','answer');e[0]['content']='[]';assert packet(e)['unique_passages']==0

def test_metadata_not_excerpt():
    e=bind(trace(),'question','answer');e[0]['content']='{"file": {"description": "summary"}}';assert packet(e)['passages'][0]['kind']=='document_metadata'

def test_dedup_provenance_and_spaces():
    e=bind(trace(),'question','answer');p=packet(e+e);assert p['unique_passages']==1;assert len(p['passages'][0]['provenance'])==2
    assert p['passages'][0]['text']=='source\nwith formatting';assert not p['exhaustive_inventory']
    assert verify_quote(p,'P001','source\nwith formatting');assert not verify_quote(p,'P001','source with formatting')
    assert not verify_quote(p,'bad','source');assert not verify_quote(p,'P001',' ')

def test_quantile_tie_rounding_lower():
    q=np.array([.5,1.5]);assert np.argmin(abs(q[:,None]-np.arange(3)[None,:]),axis=1).tolist()==[0,1]

def test_monotone_candidates_count():assert len(list(itertools.combinations_with_replacement(range(3),3)))==10

def test_one_cluster_rejected():
    with pytest.raises(ValueError):crossfit([0,1,2],[0,1,2],[0,1,2],[0,0,0])

def test_same_training_map_is_deterministic():assert monotone(np.array([0,1,2]),np.array([0,1,2]))==(0,1,2)

VECTORS=json.loads((Path(__file__).resolve().parents[1]/'results/evaluation_vectors.json').read_text())
@pytest.mark.parametrize('held',range(16))
def test_held_cluster_labels_cannot_affect_own_predictions(held):
    y=np.array([r['human'] for r in VECTORS]);a=np.array([r['A'] for r in VECTORS]);b=np.array([r['B'] for r in VECTORS]);g=np.array([r['cluster'] for r in VECTORS])
    old,soft,_=crossfit(y,a,b,g);changed=y.copy();changed[g==held]=(changed[g==held]+1)%3
    new,newsoft,_=crossfit(changed,a,b,g)
    for name in PROFILES:
        assert np.array_equal(old[name][g==held],new[name][g==held])
        assert np.allclose(soft[name][g==held],newsoft[name][g==held],atol=1e-12)

def test_cluster_row_permutation_invariance():
    y=np.array([r['human'] for r in VECTORS]);a=np.array([r['A'] for r in VECTORS]);b=np.array([r['B'] for r in VECTORS]);g=np.array([r['cluster'] for r in VECTORS])
    p,_,_=crossfit(y,a,b,g);idx=np.random.default_rng(7).permutation(len(y));pp,_,_=crossfit(y[idx],a[idx],b[idx],g[idx])
    for name in PROFILES:assert np.array_equal(p[name][idx],pp[name])

def test_bootstrap_constant_spearman_undefined():
    y=[0,0,1,1,2,2];r=paired_bootstrap(y,[2]*6,y,[0,0,1,1,2,2],20)
    assert r['metrics']['spearman']['delta'] is None;assert r['metrics']['spearman']['undefined_replicates']==20

def setup_inputs(tmp_path):
    private=tmp_path/'private';results=tmp_path/'results';private.mkdir();results.mkdir()
    (private/'blind_inputs.json').write_text(json.dumps([{'unit':'X001'},{'unit':'X002'}]))
    (private/'gold.json').write_text(json.dumps([{'unit':'X001','cluster':'x','gold':0},{'unit':'X002','cluster':'y','gold':2}]))
    (results/'selection_manifest.json').write_text(json.dumps({'n':2,'blind_inputs_sha256':hashlib.sha256((private/'blind_inputs.json').read_bytes()).hexdigest(),'gold_sha256':hashlib.sha256((private/'gold.json').read_bytes()).hexdigest()}))
    return private,results

def test_cannot_record_B_before_A(tmp_path):
    p,r=setup_inputs(tmp_path)
    with pytest.raises(ValueError):record(p,r,'B',[{'unit':'X001','score':0,'reason':'x'}])

def test_write_once_seals_and_reveal(tmp_path):
    p,r=setup_inputs(tmp_path);rows=[{'unit':'X001','score':0,'reason':'x'},{'unit':'X002','score':2,'reason':'x'}]
    record(p,r,'A',rows)
    with pytest.raises(ValueError):record(p,r,'A',rows)
    with pytest.raises(FileNotFoundError):reveal(p,r)
    record(p,r,'B',rows);v=reveal(p,r);assert len(v)==2;assert v[0]['human']==0
    (r/'B_predictions.json').write_text('[]')
    with pytest.raises(ValueError):reveal(p,r)

@pytest.mark.parametrize('row',[{'unit':'X001','score':.5,'reason':'x'}, {'unit':'unknown','score':0,'reason':'x'}, {'unit':'X001','score':True,'reason':'x'}, {'unit':'X001','score':0,'reason':''}, {'unit':'X001','score':0,'reason':'x','gold':0}])
def test_invalid_prediction_schema(tmp_path,row):
    p,r=setup_inputs(tmp_path)
    with pytest.raises(ValueError):record(p,r,'A',[row])
