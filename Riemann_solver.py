import numpy as np
from matplotlib import pyplot as plt
import scipy
GAMMA=5/3
PHI = None
f_p2p1 = lambda r,Wl,Wr,g,c1,c4,r41: np.abs( r41-r* np.power(1+((g-1)/(2*c4))*(Wl[1]-Wr[1]-(c1/g)*((r-1)/np.sqrt(1+(r-1)*((g+1)/(2*g))))),(-2*g/(g-1))))
def Riemann_exact(t,g,Wl,Wr,grid):
    """ Computes the exact solution of the Riemann problem with the state
        vectors W = [rho u P] at the left and right of the discontinuity.
        The initial discontinuity is located at x=0
        Other arguments:
            t - time
            g - gamma
            grid - grid over which to solve the problem"""
    # Solve for p2/p1
    r41 = Wl[2]/Wr[2]
    c1  = np.sqrt(g*Wr[2]/Wr[0])
    c4  = np.sqrt(g*Wl[2]/Wl[0])

    opt_fun = lambda x: f_p2p1(x,Wl,Wr,g,c1,c4,r41)
    # r21 = scipy.optimize.brentq(opt_fun, 0., 1e1, args=(), xtol=1e-10, rtol=1e-10, maxiter=100, full_output=False, disp=True)
    r21 = scipy.optimize.newton(func=opt_fun, x0=3., tol=1e-10, maxiter=50,
                                rtol=1e-10, full_output=False, disp=False)
    assert opt_fun(r21)<1e-7, 'Newton did not converge'
    # out = scipy.optimize.minimize(fun = lambda x: f_p2p1(x,Wl,Wr,g,c1,c4,r41),
                                  # x0=3, bounds=((0,np.inf),),
                                  # tol=1e-10)
    # r21 = out.x[0]

    # Compute the remainder of properties in region 2 (post-shock)
    v2 = Wr[1] + (c1/g)*(r21-1)/np.sqrt(1+((g+1)/(2*g))*(r21-1))
    c2 = c1*np.sqrt(r21*(r21+((g+1)/(g-1)))/(1+r21*((g+1)/(g-1))))
    V = Wr[1] + c1*np.sqrt(1+(r21-1)*(g+1)/(2*g))
    P2 = r21*Wr[2]
    rho2 = g*P2/np.power(c2,2)

    # Determine the properties fo the flow in region 3
    v3 = v2
    P3 = P2
    c3 = c4 + (g-1)*(Wl[1]-v3)/2
    rho3 = g*P3/np.power(c3,2)

    # Find the boundaries of each region
    x1 = V*t
    x2 = v3*t
    x3 = (v3-c3)*t
    x4 = (Wl[1]-c4)*t

    # Compute the values of the state vector in regions 1 to 4
    P = Wr[2]*(grid > x1) + P2*(grid <= x1)*(grid > x2) + P3*(grid <= x2)*(grid > x3) + Wl[2]*(grid <= x4)
    rho = Wr[0]*(grid > x1) + rho2*(grid <= x1)*(grid > x2) + rho3*(grid <= x2)*(grid > x3) + Wl[0]*(grid <= x4)
    u = Wr[1]*(grid > x1) + v2*(grid <= x1)*(grid > x2) + v3*(grid <= x2)*(grid > x3) + Wl[1]*(grid <= x4)

    # Do the same in the expansion fan
    if t!=0:
        grid_over_t = grid/t
    else:
        grid_over_t = np.sign(grid) # pour que les résultats restent bons
    u = u + (grid <= x3)*(grid > x4)*(2*((grid_over_t)+c4+(g-1)*Wl[1]/2)/(g+1))
    cfan = (grid <= x3)*(grid > x4)* (2*((grid_over_t)+c4+(g-1)*Wl[1]/2)/(g+1)-grid_over_t)
    P = P + (grid <= x3)*(grid > x4)*Wl[2]*np.power(cfan/c4, 2*g/(g-1))
    rho[np.where((grid <= x3) & (grid > x4))] = g*P[np.where((grid <= x3) & (grid > x4))] / np.power(cfan[np.where((grid <= x3) & (grid > x4))], 2)
    return rho,u,P

def JACOBIANA(W):
    '''
    Compute the jacobian matrix of primitive variable A(W)
    '''
    return np.array([[W[1],W[0],0],[0,W[1],1/W[0]],[0,GAMMA*W[2],W[1]]]) 
def Inverse_JACOBIANA(W):
    '''
    Compute the inverse jacobian matrix of primitive variable A^-1(W)
    '''
    rho,v,u,w,P=W
    a=np.sqrt(GAMMA*P/rho)
    return 1/(v**2-a**2)*np.array([[(v**2-a**2)/v,-rho,1/v],[0,v,-1/rho],[0,-rho*a**2,v]]) 
def HLLC_State_Star(W,S,SK):
    """Compute U_Star vector

    Args:
        W (_type_): primitive variable
        S (_type_): S_star
        SK (_type_): _description_

    Returns:
        _type_: _description_
    """
    rhoK,uK,vK,wK,PK=W
    EK = ((uK**2+vK**2+wK**2)/2+PK/((GAMMA-1)*rhoK))*rhoK
    UStarK = np.array([1,S,vK,wK,EK/rhoK+(S-uK)*(S+PK/(rhoK*(SK-uK)))])*rhoK*((SK-uK)/(SK-S))
    return UStarK
def decide_qk(ps,pK):
    """Compute q_k as a scaling factor of sound speed

    Args:
        ps (_type_): p_star
        pK (_type_): p_K, K=L or R

    Returns:
        _type_: _description_
    """
    if ps<=pK:
        return 1
    if ps>pK:
        return np.sqrt(1+(GAMMA+1)/2/GAMMA*(ps/pK-1))
    
def HLLC_Riemann_Solver(WL,WR,PHI=None):
    """Compute HLLC Flux

    Args:
        WL (_type_): Left state in primitive variable (rho,u,v,w,p)
        WR (_type_): Right state in primitive variable (rho,u,v,w,p)
        PHI (float, optional): Gravitational potential. Defaults to None. If consider self gravity in simulation, the formula of energy will be adjusted.

    Returns:
        np.array: F_hllc
    """
    rhoL,uL,vL,wL,PL=WL
    rhoR,uR,vR,wR,PR=WR
    #EL = (uL**2/2+PL/((GAMMA-1)*rhoL))*rhoL
    FR = W2F(WR,PHI)
    UR = W2U(WR,PHI)    
    FL = W2F(WL,PHI)
    UL = W2U(WL,PHI)
    if rhoL<=0 or PL<0 or rhoR<=0 or PR<0:
    #if PL/rhoL<0 or PR/rhoR<0 :
        Fhllcih = np.array([0,0,0,0,0])
        W_bar = (WL+WR)/2
        return Fhllcih,W_bar
    #ER = (uR**2/2+PR/((GAMMA-1)*rhoR))*rhoR
    aR=np.sqrt(GAMMA*PR/rhoR)
    aL=np.sqrt(GAMMA*PL/rhoL)
    abar = (aL+aR)/2
    rhobar = (rhoL+rhoR)/2
    pvrs = (PL+PR)/2 - (uR-uL)*rhobar*abar/2
    ps = np.max([0,pvrs])
    qL = decide_qk(ps,PL)
    qR = decide_qk(ps,PR)
    SL = uL-aL*qL
    SR = uR+aR*qR
    #SL = np.minimum(uL-aL, uR-aR)
    #SR = np.minimum(uL+aL, uR+aR)
    S = (PR-PL+rhoL*uL*(SL-uL)-rhoR*uR*(SR-uR))/(rhoL*(SL-uL)-rhoR*(SR-uR))

    USL = HLLC_State_Star(WL,S,SL)
    USR = HLLC_State_Star(WR,S,SR)

    if SL>=0:
        Fhllcih = FL
    elif SL<=0 and S>=0:
        Fhllcih = FL+SL*(USL-UL)
        #Fhllcih = W2F(U2W(USL))
    elif S<=0 and SR>=0:
        Fhllcih = FR+SR*(USR-UR)
        #Fhllcih = W2F(U2W(USR))
    elif SR<=0:
        Fhllcih = FR
    else:
        print('meet problem in HLLC')
        Fhllcih=np.array([0,0,0,0,0])
    W_bar = (U2W(USL,PHI)+U2W(USR,PHI))/2
    #Whllc = np.multiply(Inverse_JACOBIANA(W_bar),Fhllcih)
    return Fhllcih,W_bar
def U2W(U,PHI=True):
    """transform from conserved variables U to primitive variables W

    Args:
        U (ndarray): conserved variables U
        PHI (float, optional): Gravitational potential.Defaults to True. If consider self gravity in simulation, the formula of energy will be adjusted. 

    Returns:
        ndarray: primitive variables W
    """
    rho,rhou,rhov,rhow,E = U
    u = rhou/rho
    v = rhov/rho
    w= rhow/rho
    if PHI !=None:
        p = (E/rho-0.5*(u**2+v**2+w**2)-PHI/2)*((GAMMA-1)*rho)    
    else:
        p = (E/rho-0.5*(u**2+v**2+w**2))*((GAMMA-1)*rho)
    return np.array([rho,u,v,w,p])
def W2U(W,PHI=True):
    """transform from primitive variables W to conserved variables U

    Args:
        W (ndarray): primitive variables W
        PHI (float, optional): Gravitational potential.Defaults to True. If consider self gravity in simulation, the formula of energy will be adjusted. 

    Returns:
        ndarray: conserved variables U
    """
    rho,u,v,w,p = W
    if PHI !=None:
        return np.array([rho,rho*u,rho*v,rho*w,rho*(0.5*(u**2+v**2+w**2)+p/((GAMMA-1)*rho)+PHI/2)])
    else:
        return np.array([rho,rho*u,rho*v,rho*w,rho*(0.5*(u**2+v**2+w**2)+p/((GAMMA-1)*rho))])
def W2F(W,PHI=True):
    """Compute Flux from primitive variable W

    Args:
        W (ndarray): primitive variables W
        PHI (float, optional): Gravitational potential.Defaults to True. If consider self gravity in simulation, the formula of energy will be adjusted. 

    Returns:
        ndarray: Flux
    """
    rho,u,v,w,p = W
    if PHI !=None:
        E = ((u**2+v**2+w**2)/2/2+p/((GAMMA-1)*rho))*rho/2 #势能项改为除以2保证伽利略变换下能量守恒形式不变
        return np.array([rho*u,rho*u**2+p,rho*u*v,rho*u*w,u*(E+rho*PHI+p)])
    else:
        E = ((u**2+v**2+w**2)/2+p/((GAMMA-1)*rho))*rho
        return np.array([rho*u,rho*u**2+p,rho*u*v,rho*u*w,u*(E+p)])
def run(M=100,T=0.2,CFL=0.9,rhoL = 1,uL= 0.75,vL=0,wL=0,pL=1,rhoR = 0.125,uR=0,vR=0,wR=0,pR=0.1,x0 =0.3):
    """Perform simulation using Godunov scheme

    Args:
        M (int, optional): Number of pixels. Defaults to 100.
        T (float, optional): End Time. Defaults to 0.2.
        CFL (float, optional): Courant Friedrichs Lewy coefficient. Defaults to 0.9.
        rhoL (int, optional): Density of left state. Defaults to 1.
        uL (float, optional): Normal velocity of left state. Defaults to 0.75.
        vL (int, optional): Tangential velocity of left state. Defaults to 0.
        wL (int, optional): Tangential velocity of left state. Defaults to 0.
        pL (int, optional): Pressure of left state. Defaults to 1.
        rhoR (float, optional): Density of right state. Defaults to 0.125.
        uR (int, optional): Normal velocity of right state. Defaults to 0.
        vR (int, optional): Tangential velocity of right state. Defaults to 0.
        wR (int, optional): Tangential velocity of right state. Defaults to 0.
        pR (float, optional): Pressure of right state. Defaults to 0.1.
        x0 (float, optional): Position of the discontinuity. Defaults to 0.3.

    Returns:
        fig: final figure of density,velocity,pressure,energy
    """
    #M=100;dx=1/M;dt=0.01;T=0.2;
    fig,ax = plt.subplots(2,2,sharex=True)
    IC_X=np.linspace(0,1,M)
    
    #rhoL = 1;uL= 0.75;vL=0;wL=0;pL=1;rhoR = 0.125;uR=0;vR=0;wR=0;pR=0.1;x0 =0.3; #test1
    rhoL = 1;uL= -2 ;pL=0.4;vL=0;wL=0;vR=0;wR=0;rhoR=1;uR=2;pR=0.4;x0 =0.5; #test2
    
    #rhoL = 1;uL= -2  ;pL=0.4;rhoR=1;uR=2;pR=0.4; #test4
    

    """Set Initial Condition
    """
    IC_W = np.array([[rhoL,uL,vL,wL,pL]]*int(M*x0)+[[rhoR,uR,vR,wR,pR]]*(M-int(M*x0)))
    result = [[[rhoL,rhoL*uL,rhoL*vL,rhoL*wL,rhoL*(uL**2/2+pL/((GAMMA-1)*rhoL))]]*int(M*x0)+[[rhoR,rhoR*uR,rhoR*vR,rhoR*wR,rhoR*(uR**2/2+pR/((GAMMA-1)*rhoR))]]*(M-int(M*x0))]
    IC_U = np.array(result[0])
    ax[0][0].plot(IC_X-x0,IC_W.T[0],label='initial')
    ax[0][1].plot(IC_X-x0,IC_W.T[1])
    ax[1][0].plot(IC_X-x0,IC_W.T[-1]) 
    ax[1][1].plot(IC_X-x0,IC_U.T[-1])
    #for t in np.arange(0,TT-1)[:2]:
    t=0;dx=1/M
    TT = [0]
    while t<T:
        
        Old_U = result[-1]
        Old_W = U2W(np.array(Old_U).T,PHI).T
        """
        decide dt for current time.
        """
        dt = CFL*dx/np.max(np.abs(np.linalg.norm(Old_W[:,1:4],axis=1)+np.sqrt(GAMMA*Old_W[:,-1]/Old_W[:,0])))
        print('t=\r',t, end='')
        New_U = Old_U.copy()
        for i in range(M):
            """
            transmissive boundary conditions
            """
            if i == 0:
                WL = U2W(Old_U[i]  ,PHI)
                WR = U2W(Old_U[i+1],PHI)
                Wm = U2W(Old_U[i]  ,PHI)
            elif i == M-1:
                WL = U2W(Old_U[i-1],PHI)
                WR = U2W(Old_U[i],PHI)
                Wm = U2W(Old_U[i],PHI)
            else:
                Wm = U2W(Old_U[i],PHI)
                WL = U2W(Old_U[i-1],PHI)
                WR = U2W(Old_U[i+1],PHI)
            Fp,_= HLLC_Riemann_Solver(Wm,WR)
            Fm,_= HLLC_Riemann_Solver(WL,Wm)
            #Fp = CFD.HLLC_solver(W2U(Wm),W2U(WR),0)
            #Fm = CFD.HLLC_solver(W2U(WL),W2U(Wm),0)
            #result[t+1][i]=result[t][i]+dt/dx*(Fm-Fp)
            New_U[i] = Old_U[i]+dt/dx*(Fm-Fp)
        result.append(New_U)
        TT.append(t)
        t+=dt
    #np.save('./result3d',np.array(result))
    #np.save('./timeseries',np.array(TT))
    #for i in range(len(result)):
    W_final = U2W(np.array(result[-1]).T,PHI)
    ax[0][0].plot(IC_X-x0,W_final[0],label='simulation')
    ax[0][1].plot(IC_X-x0,W_final[1])
    ax[1][0].plot(IC_X-x0,W_final[-1]) 
    ax[1][1].plot(IC_X-x0,np.array(result[-1]).T[-1]) 
    #if True:
    if False:
        mesh=IC_X-x0
        mesh_exact = np.linspace(np.min(mesh), np.max(mesh), int(2e3))
        exactsol = Riemann_exact(t=T, g=GAMMA,
                                Wl=np.array([rhoL,uL,pL]),
                                Wr=np.array([rhoR,uR,pR]),
                                grid=mesh_exact)
        rho_exact = exactsol[0]
        u_exact = exactsol[1]
        P_exact = exactsol[2]
        E_exact = rho_exact*(0.5*u_exact**2+P_exact/(GAMMA-1)/rho_exact)

        ax[0][0].plot(mesh_exact, rho_exact, color='r', label='exact')
        ax[0][1].plot(mesh_exact, u_exact, color='r', label='exact')
        ax[1][0].plot(mesh_exact, P_exact, color='r', label='exact')
        ax[1][1].plot(mesh_exact, E_exact, color='r', label='exact')
    
    ax[0][0].set_xlabel('x (m)')
    ax[0][0].set_ylabel(r'$\rho$ (kg.m$^{-3}$)')
    ax[0][0].set_title('Density')
    ax[0][0].legend()
    
    
    ax[0][1].set_xlabel('x (m)')
    ax[0][1].set_ylabel(r'$u$')
    ax[0][1].set_title('Velocity (m/s)')

    
    ax[1][0].set_xlabel('x (m)')
    ax[1][0].set_ylabel('P (Pa)')
    ax[1][0].set_title('Pressure',y=0.85)
    
    
    ax[1][1].set_xlabel('x (m)')
    ax[1][1].set_ylabel('E (erg)')
    ax[1][1].set_title('Energy',y=0.85)
    plt.subplots_adjust(hspace=0,wspace=0.3)
    fig.savefig('./test23D.pdf', pad_inches=0)
    plt.show()
    return fig
if __name__=='__main__':
    run()