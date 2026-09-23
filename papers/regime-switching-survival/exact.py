import math, numpy as np
from scipy.integrate import solve_ivp
def exact_u(t,x,s,kappa,thetas,sigmas,lmbd):
    """u_y(t,x) = a_y(t) exp(-B(t) x), B=(1-e^{-kt})/k,
       a' = diag(-k th_y B + s_y^2 B^2/2) a + Q a, a(0)=1, Q = lmbd*[[-1,1],[1,-1]]"""
    th=np.array(thetas,float); ss=np.array(sigmas,float)**2
    Q=lmbd*np.array([[-1.,1.],[1.,-1.]])
    B=lambda r:(1-math.exp(-kappa*r))/kappa
    def f(r,a): b=B(r); return (-kappa*th*b+0.5*ss*b*b)*a+Q@a
    sol=solve_ivp(f,(0,t),[1.,1.],method='DOP853',rtol=1e-13,atol=1e-15)
    a=sol.y[:,-1]
    return a[s]*math.exp(-B(t)*x)
