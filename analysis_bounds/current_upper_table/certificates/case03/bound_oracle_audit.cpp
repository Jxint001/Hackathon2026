#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
using I=long long;
int main(){
  int Q,M; I den,pen; std::cin>>Q>>M>>den>>pen;
  std::vector<I> theta(32),expected(Q+1),answer(Q+1,1LL<<60);
  for(auto& z:theta)std::cin>>z;
  for(auto& z:expected)std::cin>>z;
  int W=Q+1; const I inf=1LL<<60;
  std::vector<I> f(W);
  for(int n=0;n<=Q;n++){int a=0;while(a*a<4*n)++a;f[n]=a;}
  for(int cap=0;cap<=Q;cap++){
    std::vector<I> dp(W*W,inf),nx(W*W,inf);
    for(int n=0;n<=cap;n++)
      dp[n*W+n]=(f[n]+2*n)*den-theta[0]*n+pen*std::max(0,2*(cap-n)-M);
    for(int k=1;k<32;k++){
      std::fill(nx.begin(),nx.end(),inf);
      for(int total=0;total<=Q;total++)
        for(int old=0;old<=std::min(cap,total);old++){
          I base=dp[total*W+old];if(base==inf)continue;
          for(int n=0;n<=std::min(cap,Q-total);n++){
            I v=base+(f[n]+2*std::max(0,n-old))*den-theta[k]*n;
            I& ref=nx[(total+n)*W+n];if(v<ref)ref=v;
          }
        }
      dp.swap(nx);
    }
    for(int total=0;total<=Q;total++)
      for(int n=0;n<=std::min(cap,total);n++)
        answer[total]=std::min(answer[total],dp[total*W+n]+pen*std::max(0,2*(cap-n)-M));
  }
  bool good=true;
  for(int t=0;t<=Q;t++)if(answer[t]!=expected[t]){
    std::cout<<"MISMATCH "<<t<<" got "<<answer[t]<<" expected "<<expected[t]<<'\n';good=false;
  }
  if(good)std::cout<<"PASS all "<<Q+1<<" g values, direct predecessor enumeration\n";
  std::cout<<"gq "<<answer[Q]<<'\n';
  return good?0:1;
}
