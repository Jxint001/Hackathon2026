"""Reproduce the audited M39 symmetric profile certificate without an LP solver."""
import hashlib,json,math,subprocess,tempfile
from fractions import Fraction
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[3]
a=json.loads((D/'bound_profile_spill_q163_symmetric_M39.json').read_text())
q,den=163,1630000;N,C,P,U=18432,1536,5,336;delta=Fraction(151,163)
assert a['q']==q and a['theta_den']==den and sum(a['theta_num'])==2*den
assert a['memory_for_C']==39
pen=den*delta/2;assert pen.denominator==1
inp=f'{q} 39 {den} {pen.numerator}\n'+' '.join(map(str,a['theta_num']))+'\n'+' '.join(map(str,a['g_num']))+'\n'
with tempfile.TemporaryDirectory(prefix='case05_bound_') as temp:
 binary=Path(temp)/'bound_final_audit'
 subprocess.run(['g++','-O3','-std=c++17',str(D/'strategy_sym_oracle_audit.cpp'),'-o',str(binary)],check=True)
 r=subprocess.run([str(binary)],input=inp,text=True,capture_output=True,check=True)
assert 'PASS all 164' in r.stdout
f=[math.isqrt(4*h)+(math.isqrt(4*h)**2<4*h)for h in range(U+1)]
assert max(f[h]-f[h-1]for h in range(1,U+1))<=2
INF=10**9;dp=[0]+[INF]*C
for _ in range(P):
 nd=[INF]*(C+1)
 for total in range(C+1):
  nd[total]=min(dp[total-h]+f[h]for h in range(min(total,U)+1))
 dp=nd
Dmin=12*dp[C];assert Dmin==2100
G=[Fraction(v,den)for v in a['g_num']]
raw=(N*(G[q]-41)+P*(-G[q]*(q-1)+2*sum(G[1:q])+41))/q+C-2*(1+delta)*C/q+Fraction(C-P+Dmin-2*P,q)
assert raw==Fraction(a['io_num'],a['io_den'])==Fraction(502982999519,66422500)
units=-(-raw.numerator//raw.denominator);io_time=(units*32768+59)//60
branch=(U+1)*12288;small=((N-q+P-1)//(P-1))*1024
T=min(io_time,branch,small);assert T==4135868 and units==7573
out={'status':'PASS','baseline_cycles':18876779,'q':q,'theta_num':a['theta_num'],'theta_den':den,'delta_num':delta.numerator,'delta_den':delta.denominator,'g_num':a['g_num'],'gq':str(G[q]),'C_capacity':39,'complete_chain_limit_under_branch':U,'complete_chain_input_projection_min':Dmin,'io_units_num':raw.numerator,'io_units_den':raw.denominator,'io_units_ceil':units,'io_bytes_lower_bound_under_branch':units*32768,'io_time_lower_bound_under_branch':io_time,'branch_time':branch,'small_core_time':small,'certified_time_lower_bound':T,'certified_speedup_upper_bound':18876779/T,'independent_oracle_audit':r.stdout.strip(),'graph_sha256':hashlib.sha256((ROOT/'data/case_05.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((ROOT/'data/config.txt').read_bytes()).hexdigest()}
(D/'bound_final.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='g_num'},indent=2))
