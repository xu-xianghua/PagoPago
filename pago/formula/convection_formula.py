"""
  formula for convection heat transfer
"""
#from math import sqrt, log, log10
import autograd.numpy as np

def flow_cross_single_tube(Re, Pr):
    """characteristic temperature is (Tw+Tinf)/2"""
    """characteristic length is diameter of outer tube"""
    """Tinf(15.5~982), Tw(21~1046)"""
    if Re < 4:
        C = 0.989
        n = 0.33
    elif Re < 40:
        C = 0.911
        n = 0.385
    elif Re < 4000:
        C = 0.683
        n = 0.466
    elif Re < 40000:
        C = 0.193
        n = 0.618
    else: # 40000 ~ 400000
        C = 0.0266
        n = 0.805
    return C * Re**n * Pr**0.3333333 # Nu

def flow_cross_single_tube_CB(Re, Pr):
    """by Churchill and Bemstein"""
    Nu = 0.3 + 0.62 * Re**0.5 * Pr**0.333 * (1 + (Re/282000.0)**0.625)**0.8 / (1 + (0.4/Pr)**0.6667)**0.25
    return Nu

def flow_cross_sphere(Re, Pr, mu_w, mu_inf):
    """flow cross a sphere"""
    """characteristic length is diameter"""
    """characteristic temperature is T_inf"""
    """0.71 < Pr < 380, 3.5 < Re < 76000"""
    Nu = 2 + (0.4 * Re**0.5 + 0.06 * Re**0.667) * Pr**0.4 * (mu_inf / mu_w)**0.25
    return Nu

def flow_over_plate(Re, Pr):
    if Re < 5.0e5: # laminar flow
        Nuf = 0.664 * Pr**(1.0/3) * Re**0.5
    else: # turblent flow
        Nuf = (0.037 * Re**0.8 - 871) * Pr**(1.0/3)
    return Nuf

def vertical_wall_natural_convection(Gr, Pr):
    """ Gr > 1000 """
    if Gr < 3.0e9:
        C = 0.59
        n = 0.25
    elif Gr < 2.0e10:
        C = 0.0292
        n = 0.39
    else:
        C = 0.11
        n = 0.3333
    return C * (Gr * Pr)**n

def horizontal_cylinder_natual_convection(Gr, Pr):
    """ Gr > 10000 """
    if Gr < 5.76e8:
        C = 0.48
        n = 0.25
    elif Gr < 4.65e9:
        C = 0.0445
        n = 0.37
    else:
        C = 0.1
        n = 0.3333
    return C * (Gr * Pr)**n

def horizontal_plate_natual_convection_1(Gr, Pr):
    """hot side upward, or cold side downward """
    """ 1e4 < Ra < 1e11 """
    Ra = Gr * Pr
    if Ra < 1.0e7:
        return 0.54 * Ra**0.25
    else:
        return 0.15 * Ra**0.25

def horizontal_plate_natual_convection_2(Gr, Pr):
    """hot side downward, or cold side upward """
    """ 1e5 < Ra < 1e10 """
    Ra = Gr * Pr
    return 0.27 * Ra**0.25

def sphere_natural_convection(Gr, Pr):
    """ natural convection for sphere. characteristic dimensoion is diameter of sphare
        Pr > 0.7, Ra < 1e11
        return Nu
    """
    Ra = Gr * Pr
    return 2 + 0.589 * Ra**0.25 / (1 + (0.469 / Pr)**(9.0/16))**(4.0/9)

def friction_factor_tube_flow_Flionenko(Re):
    """ (Darcy) friction factor for turbulent flow in tube, by Flionenko """
    return (1.82 * np.log10(Re) - 1.64)**(-2)

def flow_in_tube_Gnielinski(Re, Prf, Prw=-1, dl=0.):
    """ turbulent flow in circular tube, by Gnielinski
        2300 < Re < 1e6, 0.6 < Pr < 1e5
        dl: d/l
        return Nu
    """
    f = (1.82*np.log10(Re) - 1.64)**(-2)
    if Prw <= 0:
        ct = 1.
    else:
        ct = (Prf/Prw)**0.11
    t1 = f/8
    Nu = t1*(Re-1000.)*Prf*(1+(dl)**0.6667)*ct/(1+12.7*(t1**0.5)*(Prf**0.6667-1))
    return Nu

def flow_in_tube_Dittus_boelter(Re, Pr, heating=True):
    """Nu of forced convection in tube, turblence, Dittus-Boelter formula"""
    if heating is True:
        n = 0.4
    else:
        n = 0.3
    return 0.023*Re**0.8*Pr**n

def friction_factor_pipe_flow(Re_):
    """ friction factor in circle pipe """
    def laminar_flow(Re):
        if Re < 1e-120:
            Re = 1e-120
        f = 64./Re
        return f
    def turbulent_flow(Re):
        f = (1.82 * np.log10(Re) - 1.64)**(-2)
        return f
    sign = 1.0
    if Re_ < 0:
        sign = -1.0
        Re_ = -Re_
    if Re_ < 2000.:
        f = laminar_flow(Re_)
    elif Re_ < 2300.:
        f2000 = laminar_flow(2000.)
        f2300 = turbulent_flow(2300.)
        f = f2000 + (Re_-2000.)/300*(f2300 - f2000)
    else:
        f = turbulent_flow(Re_)
    return f*sign

def heat_transfer_pipe_flow(Re_, Pr, dl=0.):
    """ friction factor and Nusselt number in circle pipe
        dl: d/l
    """
    def laminar_flow(Re):
        if Re < 1e-120:
            Re = 1e-120
        f = 64./Re
        l0 = dl * 0.05 * Re * Pr
        if l0 > 1:
            Nu = 1.86*(Re*Pr*dl)**0.333
        else:
            Nu = 5. * l0 + 4.36*(1.-l0)
        return Nu, f
    def turbulent_flow(Re):
        f = (1.82 * np.log10(Re) - 1.64)**(-2)
        Nu = (f/8.)*(Re - 1000.)*Pr/(1 + 12.7*(f/8.)**0.5*(Pr**0.6667-1))*(1+(dl)**0.6667)
        return Nu, f
    if Re_ < 2000.:
        Nu, f = laminar_flow(Re_)
    elif Re_ < 2300.:
        Nu2000, f2000 = laminar_flow(2000.)
        Nu2300, f2300 = turbulent_flow(2300.)
        f = f2000 + (Re_-2000.)/300*(f2300 - f2000)
        Nu = Nu2000 + (Re_-2000.)/300*(Nu2300-Nu2000)
    else:
        Nu, f = turbulent_flow(Re_)
    return Nu, f

def friction_factor_rect_duct_laminar_developing_flow(x, Dh, ac, Re):
    """ laminar flow friction factor in the entrance region of rectangular ducts
        x: length, Dh: hydraulic diameter, ac: aspect ratio, Re: Renauld number
    """
    a = [141.97, 142.05, 142.1, 286.65]
    b = [-7.0603, -5.4166, -7.3374, 25.701]
    c = [2603, 1481, 376.69, 337.81]
    d = [1431.7, 1067.8, 800.92, 1091.5]
    e = [14364, 13177, 14010, 26415]
    f = [-220.77, -108.52, -33.894, 8.4098]
    y = lambda x,i: (a[i]+c[i]*x**0.5+e[i]*x)/(1+b[i]*x**0.5+d[i]*x+f[i]*x**1.5)
    if ac > 1:
        ac = 1./ ac
    xp = x / (Dh * Re)
    if xp > 1.:
        xp = 1.
    if ac > 0.5:
        fapp1 = y(xp, 0)
        fapp2 = y(xp, 1)
        return (fapp2 + 2*(ac - 0.5)*(fapp1 - fapp2))/Re
    if ac > 0.2:
        fapp1 = y(xp, 1)
        fapp2 = y(xp, 2)
        return (fapp2 + 3.333*(ac - 0.2)*(fapp1 - fapp2))/Re
    if ac > 0.1:
        fapp1 = y(xp, 2)
        fapp2 = y(xp, 3)
        return (fapp2 + 10*(ac - 0.1)*(fapp1 - fapp2))/Re
    return y(xp, 3)/Re                  # ac <= 0.1
    
def Po_rect_duct_laminar_developed_flow(ac):
    """ Poiseuille number(Fanning friction factor times Re) for fully developed laminar flow rectangular ducts
        ac: aspect ratio
    """
    if ac < 1.:
        ac = 1./ac
    if ac < 1.5:
        return 14.23
    if ac < 2.5:
        return 15.55
    if ac < 3.5:
        return 17.09
    if ac < 5:
        return 18.23
    if ac < 7:
        return 19.7
    if ac < 10:
        return 20.58
    return 24.


def Nu_rect_duct_laminar_developing_flow(x, Dh, ac, Re, Pr):
    """ laminar flow convection heat transfer in the entrance region of rectangular ducts
        x: length, Dh: hydraulic diameter, ac: aspect ratio, Re: Renauld number, Pr: Prantle number
    """
    a = [36.736, 30.354, 31.297, 28.315, 6.7702, 9.1319]
    b = [2254, 1875.4, 2131.3, 3049, -3.1702, -3.7531]
    c = [17559, 13842, 14867, 27038, 0.4187, 0.48222]
    d = [66172, 154970, 144550, 472520, 2.1555, 2.5622]
    e = [555480, 783440, 622440, 1783300, 2.76e-6, 5.16e-6]
    f = [1212.6, -8015.1, -13297, -35714]

    y1 = lambda x: (a[0]+c[0]*x+e[0]*x**2)/(1+b[0]*x+d[0]*x**2+f[0]*x**3)
    y2 = lambda x: (a[1]+c[1]*x+e[1]*x**2)/(1+b[1]*x+d[1]*x**2+f[1]*x**3)
    y3 = lambda x: (a[2]+c[2]*x+e[2]*x**2)/(1+b[2]*x+d[2]*x**2+f[2]*x**3)
    y4 = lambda x: (a[3]+c[3]*x+e[3]*x**2)/(1+b[3]*x+d[3]*x**2+f[3]*x**3)
    y5 = lambda x: a[4]+b[4]*x+c[4]*np.log(x)**2+d[4]*np.log(x)+e[4]*x**-1.5
    y6 = lambda x: a[5]+b[5]*x+c[5]*np.log(x)**2+d[5]*np.log(x)+e[5]*x**-1.5

    if ac > 1:
        ac = 1./ ac
    xp = 1. * x / (Dh * Re * Pr)
    if xp > 1.:
        xp = 1.
    if ac > 0.5:
        nu1 = y4(xp)
        nu2 = y5(xp)
        return nu1 + 2*(ac - 0.5)*(nu2 - nu1)
    if ac > 0.333:
        nu1 = y3(xp)
        nu2 = y4(xp)
        return nu1 + 5.988*(ac - 0.333)*(nu2 - nu1)
    if ac > 0.25:
        nu1 = y2(xp)
        nu2 = y3(xp)
        return nu1 + 12.05*(ac - 0.25)*(nu2 - nu1)
    if ac > 0.1:
        nu1 = y1(xp)
        nu2 = y2(xp)
        return nu1 + 6.6667*(ac - 0.1)*(nu2 - nu1)
    return y1(xp)                  # ac <= 0.1

