from ast import Constant
import random
from dolfin import *
import dolfin
import numpy as np

#mesh_xdmf = XDMFFile(MPI.comm_world, "1_element_NS.xdmf")
#mesh = Mesh()
#mesh_xdmf.read(mesh)

ele=16
length=1
mesh=IntervalMesh(ele,0,length)

coordinates = mesh.coordinates()
print("Vertex vs Coordinates:")
for i in range(len(coordinates)): 
    
    print(i, "\t", coordinates[i])


P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
P2 = FiniteElement("CG", mesh.ufl_cell(), 1)



W = FunctionSpace(mesh, P1*P2)


# Boundaries

def uL_boundary(x):
    return x[0] <= DOLFIN_EPS


def uR_boundary(x):
    return x[0] >= (length - DOLFIN_EPS)


#################################### test section

V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)



uL = Constant(10)
uR = Constant(0)
bc1 = DirichletBC(V, uL, uL_boundary)
bc2 = DirichletBC(V, uR, uR_boundary)
bcs = [bc1,bc2]

f = Constant(0)
a = inner(grad(u), grad(v))*dx
L = f*v*dx 

u = Function(V)
solve(a == L, u, bcs)

print(u.vector()[:])

#####################################

r = 0.1 # m

A0 = np.pi*r**2 # m^2

rho = 1060 # kg*m^-3

mu=0.0035 # Pa*s

alpha = 1.3333

Beta = 5*10**4 #Pa*m #np.sqrt(np.pi) * h0 * E / (1-nu**2) 

k = 2 * np.pi * alpha * mu / ( rho * (alpha-1) )  

#namda = 5*10**6 #Pa/M^2

area_point=np.linspace(0,ele*2+1,ele*2+2)[0::2]
velocity_point=np.linspace(0,ele*2+1,ele*2+2)[1::2]



du = TrialFunction(W)
H = TestFunction(W)

u = Function(W)



for area in area_point:
    u.vector()[area]=0.03
#u.vector()[:]=0.1
print(u.vector()[:])


da, dq = split(du)
A, Q = split(u)
a, q = split(H)




Pin = 2100 ####### Pressure inlet ####################

Pout = 0

Ain = (Pin*A0/Beta+np.sqrt(A0))**2 # BC area in

Aout = (Pout*A0/Beta+np.sqrt(A0))**2 # BC area out

print(Ain,Aout)

f=Constant(0)   #Expression("x[0]*0", degree=1)


F1 =Q.dx(0)*a*dx #- f*a*dx #+ Q*a*ds


F2 = alpha*(Q**2/A).dx(0)*q*dx + Beta/(3*rho*A0)*(3/2*A**(1/2))*A.dx(0)*q*dx + k*Q/A*q*dx\
    # + (alpha*Q**2/A)*q*ds+Beta*A**(3/2)/(3*rho*A0)*q*ds

F = F1 + F2



Jac = derivative(F, u, du)

bcA1 = DirichletBC(W.sub(0), Ain, uL_boundary) # BC for Inlet Area
bcA2 = DirichletBC(W.sub(0), Aout, uR_boundary) # BC for Outlet Area

bcQ1 = DirichletBC(W.sub(1), 0.3, uL_boundary) # BC for Inlet Flowrate
bcQ2 = DirichletBC(W.sub(1), 0.3, uR_boundary) # BC for Outlet Flowrate

bcs = [bcA1,bcA2]#,bcQ2]


bcs_nt = [DirichletBC(W.sub(0), Constant(0.0), 'near(x[0], 0)'),
            DirichletBC(W.sub(0), Constant(0.0), 'near(x[0], 1)')]





duu = u.copy(deepcopy=True)
duu.vector()[:] = 0.0
rel_tol = 1e-8
abs_tol = 1e-8
maxiter = 50
rel_res = 1e9
res = 1e9
it = 0


A, b = assemble_system(Jac, -F, bcs, \
            form_compiler_parameters={"representation":"uflacs"}\
                    )

# Compute Absolute error
resid0 = b.norm("l2")

# Compute Relative  error
rel_res = b.norm("l2")/resid0
res = resid0

print ("Iteration: %d, Residual: %.3e, Relative residual: %.3e" %(it, res, rel_res))
solve(A, u.vector(), b)

while (rel_res > rel_tol and res > abs_tol) and it < maxiter:

        it += 1
        B = assemble(F, \
            form_compiler_parameters={"representation":"uflacs"}\
            )
        for bc in bcs_nt:
            bc.apply(B)

        rel_res = B.norm("l2")/resid0
        res = B.norm("l2")
        print ("Iteration: %d, Residual: %.3e, Relative residual: %.3e" %(it, res, rel_res))


        A, b = assemble_system(Jac, -F, bcs_nt, \
                form_compiler_parameters={"representation":"uflacs"}\
                )
        solve(A, duu.vector(), b)
        u.vector().axpy(1.0, duu.vector())

        if it==maxiter:
            stop

print(u.vector()[:])

a1,a2=u.split() 

test = FunctionSpace(mesh, P1) 


a_P1 = project(a1, test)
a_nodal_values = a_P1.vector()[:]
print(a_nodal_values)
print(np.shape(a_nodal_values))


q_P1 = project(a2, test)
q_nodal_values = q_P1.vector()[:]
print(q_nodal_values)
print(np.shape(q_nodal_values))

coor = mesh.coordinates()

print('Area:')  # Print area value at each node          
for i in range(len(a_nodal_values)):
    print (coor[i][0], a_nodal_values[i])

print('flow rate:') # Print flowrate value at each node          
for i in range(len(q_nodal_values)):
    print (coor[i][0], q_nodal_values[i])

pressure_data = Beta/A0*(np.sqrt(a_nodal_values)-np.sqrt(A0))
print(pressure_data)


file = File('NS_steady_area_dp'+str(Pin)+'.pvd')
file << a_P1

np.savetxt("NS_area_dp"+str(Pin)+".txt",a_nodal_values,delimiter=",")
np.savetxt("NS_pressure_dp"+str(Pin)+".txt",pressure_data,delimiter=",") 
np.savetxt("NS_flowrate_dp"+str(Pin)+".txt",q_nodal_values,delimiter=",") 



