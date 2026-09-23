import math
from scipy.integrate import quad
class Corrected:
    """Survival u_s(t,x)=E[exp(-int_0^t x)] for dx=k(th_s-x)dt+sig_s dW, two regimes switching at rate lam each way.
    gbar(r)=-k thbar B+ssbar B^2/2, gtil(r)=-k thtil B+sstil B^2/2, B=(1-e^{-kr})/k, eps=1/lam.
    Outer expansion: u_s = e^{-Bx} exp(int gbar + eps/2 I - eps^2 gtil^2/8) (1 +/- (eps gtil/2 - eps^2 gtil'/4)), I=int_0^t gtil^2.
    (+ for s=0, the regime with theta_1)."""
    def __init__(s,kappa,thetas,sigmas,lmbd):
        s.k=kappa; s.lam=lmbd; s.eps=1/lmbd
        s.thb=(thetas[0]+thetas[1])/2; s.tht=(thetas[0]-thetas[1])/2
        s.ssb=(sigmas[0]**2+sigmas[1]**2)/2; s.sst=(sigmas[0]**2-sigmas[1]**2)/2
    def B(s,r): return (1-math.exp(-s.k*r))/s.k
    def gbar(s,r): b=s.B(r); return -s.k*s.thb*b+0.5*s.ssb*b*b
    def gtil(s,r): b=s.B(r); return -s.k*s.tht*b+0.5*s.sst*b*b
    def gtil_p(s,r): b=s.B(r); bp=math.exp(-s.k*r); return -s.k*s.tht*bp+s.sst*b*bp
    def parts(s,t):
        G=quad(s.gbar,0,t,epsabs=1e-14,epsrel=1e-13)[0]; I=quad(lambda r:s.gtil(r)**2,0,t,epsabs=1e-16,epsrel=1e-13)[0]
        return G,I,s.gtil(t),s.gtil_p(t)
    def series(s,t,x,sgn):
        G,I,g,gp=s.parts(t); u0=math.exp(G-s.B(t)*x); e=s.eps
        return [u0, u0*e*I/2, u0*e*sgn*g/2, u0*e*e*(I*I/8-g*g/8), u0*e*e*sgn*(I*g/4-gp/4)]
    def u_exp(s,t,x,sgn):
        G,I,g,gp=s.parts(t); e=s.eps
        return math.exp(G-s.B(t)*x+e*I/2-e*e*g*g/8)*(1+sgn*(e*g/2-e*e*gp/4))
