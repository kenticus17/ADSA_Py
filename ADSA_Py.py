import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.misc import derivative as der
from scipy.optimize import minimize
from scipy.integrate import quad
from random import randint

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


# E_BACKUP_GLOBALS
q_global = np.empty((50, 5))

π = np.pi
def update_q(q, err):
    global q_global
    a = q_global[1:]
    q = np.append(q, err)
    b = np.array(((q),))
    q_global = np.append(a, b, axis=0)

# Integration of the Laplace Function for the Calculated Curve u = u(s)
def dx(φ):
    return np.cos(φ)

def dz(φ): 
    return np.sin(φ)

def dφ(φ, x, z, b, c):
    if  x == 0:
        return 1/b
    return 2*b + c*z - np.sin(φ)/x

def RK(zmax, b, c, Δs = 1e-4, maxit = 1e8):
    maxit = int(maxit)
    sset = np.zeros(1)
    xset = np.zeros(1)
    zset = np.zeros(1)
    φset = np.zeros(1)
    def stepCalc(Δs, i):
        Δφ1 = Δs*dφ(φset[i], xset[i], zset[i], b, c)
        Δx1 = Δs*dx(φset[i])
        Δz1 = Δs*dz(φset[i])
        
        Δφ2 = Δs*dφ(φset[i]+0.5*Δφ1, xset[i]+0.5*Δx1, zset[i]+0.5*Δz1, b, c)
        Δx2 = Δs*dx(φset[i]+0.5*Δφ1)
        Δz2 = Δs*dz(φset[i]+0.5*Δφ1)
        
        Δφ3 = Δs*dφ(φset[i]+0.5*Δφ2, xset[i]+0.5*Δx2, zset[i]+0.5*Δz2, b, c)
        Δx3 = Δs*dx(φset[i]+0.5*Δφ2)
        Δz3 = Δs*dz(φset[i]+0.5*Δφ2)
        
        Δφ4 = Δs*dφ(φset[i]+Δφ3, xset[i]+Δx3, zset[i]+Δz3, b, c)
        Δx4 = Δs*dx(φset[i]+Δφ3)
        Δz4 = Δs*dz(φset[i]+Δφ3)
        
        φadd = φset[i] + (Δφ1+ Δφ2+Δφ2 + Δφ3+Δφ3 + Δφ4)/6
        xadd = xset[i] + (Δx1+ Δx2+Δx2 + Δx3+Δx3 + Δx4)/6
        zadd = zset[i] + (Δz1+ Δz2+Δz2 + Δz3+Δz3 + Δz4)/6
        sadd = sset[i] + Δs
        return φadd, xadd, zadd, sadd
    def adjustStep(Δs, i, tol = .1):
        φ1, x1, z1, s1 = stepCalc(Δs, i)
        φ2, x2, z2, s2 = stepCalc(Δs*2, i)
        if abs(x2-x1)<tol and abs(z1-z2)<tol:
            Δs = Δs*2
        else:
            if Δs>=(2e-4):
                Δs = Δs*0.5
        return Δs
    
    for i in range(maxit):
        φadd, xadd, zadd, sadd = stepCalc(Δs, i)
        φset = np.append(φset, φadd)
        xset = np.append(xset, xadd)
        zset = np.append(zset, zadd)
        sset = np.append(sset, sadd)
        if zset[i+1] > zmax:
            return φset, xset, zset, sset
        Δs = adjustStep(Δs, i)
    return "Max itteration reached on RK integration."
    
        
    

# Error function to sum the distances between the calculated curve and the 
# experimental points.

def E(q, U, zmax, Sessile):
#     Unpack the optimization parameter q and experimental input.
    global q_global
    x0 = q[0]
    z0 = q[1]
    b  = abs(q[2])*1e-3      #ppt
    c  = q[3]*1e-6      #ppm

        
    if Sessile==True:
        c = abs(c)
    else:
        c = -abs(c)
    
    N  = len(U)
    weight = np.ones(N)
#     weight[0:5]  = .1
#     weight[-5:] = .1
    
#     Calculate theoretical curve
    φset, xset, zset, sset = RK(zmax, b, c)
    x = CubicSpline(sset, xset)
    z = CubicSpline(sset, zset)
    
    def initiallize_α(U, x0, z0, αguess=0.01):
        def slopef(α0, x0, z0):
            UX0 = Xshift(U.transpose()[0], U.transpose()[1], α0, x0, z0)
            UZ0 = Zshift(U.transpose()[0], U.transpose()[1], α0, z0, x0)
            x0 = UX0[0]
            x1 = UX0[-1]
            y0 = UZ0[0]
            y1 = UZ0[-1]
            return (y1-y0)/(x1-x0)
        return fsolve(lambda x: slopef(x, x0, z0), αguess)[0]
    
    α = initiallize_α(U, x0, z0)

    UX = Xshift(U.transpose()[0], U.transpose()[1], α, x0, z0)
    UZ = Zshift(U.transpose()[0], U.transpose()[1], α, z0, x0)
    Ushift = np.empty((len(UX),2))
    for i in range(len(UX)):
        Ushift[i] = (UX[i], UZ[i])
    
#     Define an s dependent equation for the distance between the 
# theoretical curve and experimental points.

    def e(s, X, Z):
        if X<=0:
            ex = -x(s) - X
        else:
            ex = x(s) - X
        ez = z(s) - Z
        e  = 0.5*(ex*ex + ez*ez)
        return e

#     Iterate to find s value normal to experimental point.
    def normal_s(X, Z, imax=1000, stol=0.1):
        sn0 = np.sqrt(X*X + Z*Z)
        f = lambda v: der(lambda var: e(var, X, Z), v, dx=0.1)
        sn = fsolve(f, sn0)
        return sn
        
#     Sum the normal distance betweeen theoretical curve and
# the experimental points. 
    eset = np.empty(N)
    for i in range(N):
        X = Ushift[i][0]
        Z = Ushift[i][1]
        sn = normal_s(X, Z)
        eset[i] = e(sn, X, Z)
    
    eRange = 2 * np.std(eset)
    weight = np.where(eset < eRange, 1, 0)
    if sum(weight)/len(weight)<0.95:
        weight = np.ones(len(weight))
    if check_consecutive_zeros(weight):
        weight = np.ones(len(weight))
    err = sum(weight*eset)

    update_q(q, err)
    return err

def Zshift(X, Z, α, z0, x0):
    Xtr = X - x0
    Ztr = Z - z0
    Zr = Xtr*np.sin(α) + Ztr*np.cos(α)
    return Zr

def Xshift(X, Z, α, x0, z0):
    Xtr = X - x0
    Ztr = Z - z0
    Xr = Xtr*np.cos(α) - Ztr*np.sin(α)
    return Xr

def check_consecutive_zeros(arr, x=2):
    # Initialize a counter for consecutive zeros
    zero_count = 0
    
    # Loop over the array and count consecutive zeros
    for i in range(len(arr)):
        if arr[i] == 0:
            zero_count += 1
        else:
            zero_count = 0
        if zero_count >= x:
            return True
    
    return False

def initiallize_q0(U, zmax, Sessile, midSliceFactor = 0.1, nrad=50, cguess=7.09):
#     print('initializing parameters')
    α = 0
    N = len(U)
    
#     Apex Point initial value
    lowSlice = int(N*0.5-N*midSliceFactor)
    highSlice = int(N*0.5+N*midSliceFactor)
    nSlice = len(U[lowSlice:highSlice])
    x = np.empty(nSlice)
    z = np.empty(nSlice)
    for i in range(nSlice):
        x[i] = U[lowSlice+i][0]
        z[i] = U[lowSlice+i][1]
        
    p = np.polyfit(x, z, 2)
    def f(y):    
        return np.polyval(p, y)
    def df(y):
        return der(f, y, dx=1e-5)
    x0 = fsolve(df, U[int(0.5*N)][0])
    z0 = f(x0)
    R1 = 0
    for i in range(nrad):
        ri1 = randint(lowSlice, highSlice)
        X1 = U[ri1][0]
        Z1 = f(X1)
        ri2 = randint(lowSlice, highSlice)
        if ri1==ri2:
            ri2 = ri2+2
        X2 = U[ri2][0]
        Z2 = f(X2)
        a1 = df(X1)
        a2 = df(X2)
        
        r1 = (a2*(Z2-Z1)+(X2-X1))*np.sqrt(a1*a1+1)/(a1-a2)
        r2 = (a1*(Z2-Z1)+(X2-X1))*np.sqrt(a2*a2+1)/(a1-a2)
        R1 = r1+r2+R1
    R1 = -0.5*R1/nrad
    b  = np.array([1/R1*1e3,])
    
    U10 = GetSpacedElements(U, 10)
    
    ans  = minimize(lambda a: E([x0, z0, b, a], U10, zmax, Sessile), cguess, 
                    method='BFGS', options={'gtol': 1})
    c = ans.x
    q = np.array([x0, z0, b, c])
    return q

def Physical_Results(U, q, Sessile):
    x0 = q[0]
    z0 = q[1]
    b  = abs(q[2])*1e-3      #ppt
    c  = q[3]*1e-6      #ppm

    if Sessile==True:
        c = abs(c)
    else:
        c = -abs(c)
    def initiallize_α(U, x0, z0, αguess=0.01):
        def slopef(α0, x0, z0):
            UX0 = Xshift(U.transpose()[0], U.transpose()[1], α0, x0, z0)
            UZ0 = Zshift(U.transpose()[0], U.transpose()[1], α0, z0, x0)
            x0 = UX0[0]
            x1 = UX0[-1]
            y0 = UZ0[0]
            y1 = UZ0[-1]
            return (y1-y0)/(x1-x0)
        return fsolve(lambda x: slopef(x, x0, z0), αguess)[0]
    
    α = initiallize_α(U, x0, z0)
    UX = Xshift(U.transpose()[0], U.transpose()[1], α, x0, z0)
    UZ = Zshift(U.transpose()[0], U.transpose()[1], α, z0, x0)
    Ushift = np.empty((len(UX),2))
    for i in range(len(UX)):
        Ushift[i] = (UX[i], UZ[i])
    
    R0 = 1/b
    H  = np.max(UZ)-np.min(UZ)
    
    P1  = Ushift[0]
    P2  = Ushift[-1]
    Zprime = (P1[1]+P2[1])*0.5
    
    #     Calculate theoretical curve
    φset, xset, zset, sset = RK(Zprime, b, c)
    x   = CubicSpline(sset, xset)
    z   = CubicSpline(sset, zset)
    
    xfz = CubicSpline(np.sort(abs(zset)), xset)
    
    Xprime = xfz(Zprime)
    
    CA = π - np.arctan(-0.01/(xfz(Zprime)-xfz(Zprime-0.01)))
    
    Xmax   = np.max(Ushift.transpose()[0])
    indZ   = np.where(Ushift.transpose()[0]==Xmax)[0][0]
    Zguess = Ushift[indZ][1]
    Zmer   = fsolve(lambda var: der(xfz, var), Zguess)
    Xmer   = xfz(Zmer)
    
    A = lambda zvar: π*xfz(zvar)*xfz(zvar)
    vol = quad(lambda zvar: A(zvar), 0, np.max(zset))[0]
    SA = A(np.max(zset))
    
    return vol, CA, c, R0, H, SA

def GetSpacedElements(array, numElems):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out

# Input Data must only have points of drop edge location.
# Points ordered from left triple point, 
# around the surface clockwise, to the right
# triple point.

def ADSA(Data, Sessile):
    global q_global

    Nrow = len(Data)
    U = np.empty((Nrow, 2))
    zmax = 0
    zmin = 1e10
    for i in range(Nrow):
        x      = Data[i][0]
        z      = Data[i][1]
        U[i] = [x, z]
        if z>zmax:
            zmax = z
        if z<zmin:
            zmin = z
        zmax = zmax-zmin
    q = initiallize_q0(U, zmax, Sessile, midSliceFactor = 0.1, nrad=50)
    try:
        ans = minimize(lambda a: E(a, U, zmax, Sessile), q, 
                   method='BFGS', tol=1, options={'gtol': 10})
        vol, CA, c, R0, H, SA = Physical_Results(U, ans.x, resultFile, count, Sessile)
    except TypeError:
        print("Warning, Backup used test q values in E function.")
        i = np.argmin(q_global[:,-1])
        ans = q_global[i][:5]
        vol, CA, c, R0, H, SA = Physical_Results(U, ans, Sessile)
    return vol, CA, c, R0, H, SA
    
