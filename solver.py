import numpy as np
import math as m
import matplotlib.pyplot as plt
import polynomial_tools as pt

one = True

Nb = 4
gauss_points_num = 8
master = [0,1]
Npts = 100 # for plotting
# hermite cubic polynomials
coeffs = np.array([[1,0,-3,2],[0,1,-2,1],[0,0,3,-2],[0,0,-1,1]])

L_const = 2.54*100

s_const = 76.2
sigma0_const = 1e6
E_const = 20.7e9
I_const = 249739
d_const = 4.572*100
c_const = 0.03

A_const = s_const*sigma0_const

B_const = np.log(10)/(c_const*d_const)
K_const = A_const/(E_const*I_const)

print(A_const, B_const, K_const)

def fem(L, NEL):

    # taylor expansion of e^x
    def exp(x):
        return 1 + (-B_const*x) + (-B_const*x)**2/2 + (-B_const*x)**3/6 + (-B_const*x)**4/24

    # indicator function
    def chi(e,x):
        return 1 if x > xs[e] and x <= xs[e+1] else 0

    # change of variable
    def xi(x, xe):
        return (x - xe[0]) / (xe[1] - xe[0])

    # dot product between coeffs u and basis functions
    def get_u(u, xi):
        return u[0]*N[0](xi) + u[1]*N[1](xi) + u[2]*N[2](xi) + u[3]*N[3](xi)

    # constructs truncated force term
    def construct_f(us):
        u_local = np.zeros((NEL, Nb))
        for i in range(NEL):
            u_local[i] = us[2*i:2*i+4]

        f_local = np.zeros((NEL, Nb))
        for e in range(NEL):
            for j in range(4):
                f_local[e,j] = ls[e]*pt.gauss(lambda x: N[j](x) * exp(get_u(u_local[e],x)), master[0], master[1], gauss_points_num)

        f_global = np.zeros(len(us))
        for i in range(NEL):
            f_global[2*i:2*i+Nb] += f_local[i]

        return f_global

    # assemble u(x) from u_i coefficients
    def u_fun(u_locals, x):
        ans = 0
        for e in range(NEL):
            sm = 0
            for j in range(Nb):
                sm += u_locals[e,j] * N[j](xi(x,es[e]))
            sm *= chi(e,x)
            ans += sm
        return ans

    def get_q(u_locals, es, d2_N, d3_N):
        q = np.zeros(2)
        e = 0
        for j in range(Nb):
            q[0] += u_locals[e,j] * d2_N[j](xi(0,es[e]))
            q[1] += u_locals[e,j] * d3_N[j](xi(0,es[e]))
        q/=100
        return q

    # basis functions on the master element
    N = pt.get_basis_functions(coeffs)
    d2_N = pt.get_basis_derivatives(coeffs, 2)
    d3_N = pt.get_basis_derivatives(coeffs, 3)

    # global nodes
    xs = np.arange(0,L+.1,L/NEL)

    # mesh sizes
    ls = np.empty(NEL)
    for i in range(NEL):
        ls[i] = xs[i+1] - xs[i]

    # element nodes
    es = np.empty((NEL,2))
    for i in range(len(es)):
        es[i] = [xs[i],xs[i+1]]

    # local stiffness and mass matrices
    Kl = np.empty([NEL,Nb,Nb], float)
    Ml = np.empty([NEL,Nb,Nb], float)
    for k in range(NEL):
        for i in range(Nb):
            for j in range(Nb):
                Kl[k][i][j] = 1 / (ls[k])**3 * pt.gauss(lambda x: d2_N[i](x) * d2_N[j](x), master[0], master[1], gauss_points_num)
                Ml[k][i][j] =      ls[k]     * pt.gauss(lambda x:    N[i](x) *    N[j](x), master[0], master[1], gauss_points_num)

    # Global matrices and force term
    Kn = (NEL+1)*2
    K, M = np.zeros((Kn,Kn)), np.zeros((Kn,Kn))
    f = np.zeros(Kn)
    for i in range(NEL):
        K[2*i:2*i+Nb,2*i:2*i+Nb] += Kl[i]
        M[2*i:2*i+Nb,2*i:2*i+Nb] += Ml[i]

    for r in range(len(K)):
        for c in range(len(K[0])):
            print('%5d ' % (K[r,c]*1e8), end='')
        print()

    # Use fixed point iteration for u
    # Start with the initial guess
    us = np.zeros(len(K))
    niter = 10
    for i in range(niter):
        force = K_const*construct_f(us)
        us[2:] = np.linalg.solve(K[2:,2:], force[2:])

    # local u coefficients
    u_locals = np.empty((NEL, Nb))
    for i in range(NEL):
        u_locals[i] = us[2*i: 2*i+Nb]

    # producing piecewise polynomial approximation
    dx = L/Npts
    xxxs = np.arange(0,L+dx/2,dx)
    u_vals = np.empty(len(xxxs))
    for i in range(len(xxxs)):
        u_vals[i] = u_fun(u_locals, xxxs[i])

    q = get_q(u_locals, es, d2_N, d3_N)

    # 64|  3.087435924293E-04  |  -2.777854642124E-06  |

    return xxxs, u_vals, q

if one:
    NELs = [1]
else:
    NELs = [1,2,4,8,16,32,64]

errs = np.empty(len(NELs)-1)
uses = np.empty((len(NELs), Npts+1))
q_errs = np.empty((len(NELs)-1,2))
qs = np.empty((len(NELs),2))

string = "NEL|        u_xx(0)        |        u_xxx(0)       |"
print(string)
print("-"*len(string))
for i, nel in enumerate(NELs):
    xs, us, q = fem(L_const, nel)
    print("%3s|  %.12E  |  %.12E  |" % (nel, q[0], q[1]))
    uses[i,:] = us
    qs[i,:] = q
    plt.plot(xs/100, us/100, 'r--', label="NEL = {}".format(nel))
    plt.legend()
    plt.title('L = {} m'.format(L_const/100))
    plt.xlabel('x (m)')
    plt.ylabel('the beam deformation (m)')

    plt.xlim(0,3)
    plt.ylim(0,0.05)
    plt.xticks(np.arange(0, 3.25, step=0.5))
    plt.yticks(np.arange(0, 0.05+0.0025, step=0.005))
plt.show()

for i in range(len(NELs)-1):
    errs[i] = np.linalg.norm(uses[i+1,:] - uses[i,:])/np.linalg.norm(uses[i+1,:])
    q_errs[i,0] = abs(qs[i+1,0]-qs[i,0])/abs(qs[i+1,0])
    q_errs[i,1] = abs(qs[i+1,1]-qs[i,1])/abs(qs[i+1,1])

if not one:
    ns = np.arange(1,len(NELs),1)
    plt.plot(ns, errs, label="u err")
    plt.plot(ns, q_errs[:,0], label="u_xx(0) err")
    plt.plot(ns, q_errs[:,1], label="u_xxx(0) err")
    plt.legend()
    plt.xlabel('log of NEL')
    plt.ylabel('log of normalied relative l2-error')
    plt.yscale('log')
    plt.title('Error')
    plt.show()
