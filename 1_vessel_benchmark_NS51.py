from ast import Constant
import random
from dolfin import *
import dolfin
import numpy as np

# Class for interfacing with the Newton solver
class NS_self(NonlinearProblem):
    def __init__(self, L, a, Bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = Bcs
        
    
    def F(self, b, x):
        assemble(self.L, tensor=b)
        [bc.apply(b) for bc in bcs] 

        
    def J(self, A, x):
        assemble(self.a, tensor=A)
        
        [bc.apply(A) for bc in self.bcs]


#mesh_xdmf = XDMFFile(MPI.comm_world, "1_element_NS.xdmf")
#mesh = Mesh()
#mesh_xdmf.read(mesh)

ele=4 #Number of elements
mesh=IntervalMesh(ele,0,1.26)

#define points
mesh_points=mesh.coordinates()
points = mesh_points
#for e, entity in enumerate(points):
#    print(e, entity)
point_sort=points.tolist()
print(mesh.coordinates())



Pin=[]
with open('Pin_com_carotid.txt') as f:
    for line in f:

        Pin.append([elt.strip() for elt in line.split(',')])
Pin = np.asfarray(Pin,float)
Pin = Pin*7.5



########################## Read Input data

Pout=[]
with open('Pout_com_carotid.txt') as f:
    for line in f:

        Pout.append([elt.strip() for elt in line.split(',')])
Pout = np.asfarray(Pout,float)
Pout = Pout*7.5

Qout=[]
with open('Qout_com_carotid.txt') as f:
    for line in f:

        Qout.append([elt.strip() for elt in line.split(',')])
Qout = np.asfarray(Qout,float)
###########


# Define the timesteps for based on the size of input data
size=np.shape(Qout)[0]











# CG 1 function space

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
P2 = VectorElement("CG", mesh.ufl_cell(), 1)

W = FunctionSpace(mesh, P1*P1)

# delta t
dt=1.1/size


# Boundaries
def uL_boundary(x, on_boundary):
    tol = 1e-14
    return on_boundary and near(x[0], 0, tol) 


def uR_boundary(x, on_boundary):
    tol = 1e-14
    return on_boundary and near(x[0], 1.26, tol)





# Constant parameters
E=5250.43 #mmHg

h=0.3 #mm

beta=4/3*(np.sqrt(np.pi)*E*h)

rho=0.00106 # blood density g/mm^3

rref=3 # reference radius mm

Aref= np.pi*rref**2 #reference Area mm^2


Pext=0 # external pressure mmHg

Pd=82.0042321 # diastolic pressure mmHg

Ad=Aref # diastolic area 

ci=2 #radial coordinate

mu=0.0000300024630338264 #blood viscosity

'''
E=700

h=0.03 #mm

beta=4/3*(np.sqrt(np.pi)*E*h)

rho=1.06

rref=0.3

Aref= np.pi*rref**2

Kr=1.0

Pext=0

pd=10.933

Ad=Aref

ci=2

mu=4e-6
'''






f= -2 * (ci+2) * mu * np.pi / rho 


area_point=np.linspace(0,ele*2+1,ele*2+2)[0::2]
velocity_point=np.linspace(0,ele*2+1,ele*2+2)[1::2]





du = TrialFunction(W)
H = TestFunction(W)

u=Function(W)
u0=Function(W)

print(u.vector()[:])

for area in area_point:
    u0.vector()[area]=22

#for area in velocity_point:
#    u0.vector()[area]=1
#u0.vector()[:]=1
u.vector()[:]=u0.vector()[:]
print(u.vector()[:])



da, dq = split(du)
A, Q = split(u)
a, q = split(H)
A0, Q0 = split(u0)




F1 = A*a*dx - A0*a*dx - dt*inner( Q, grad(a)[0] )*dx
#F1 = A*a*dx - A0*a*dx + dt*dot(grad(A),grad(a))*dx

F2 = Q*q*dx - Q0*q*dx - dt*inner(  Q**2/A+beta/(3*rho*Ad)*A**(3/2),grad(q)[0]) * dx - dt*f*Q/A*q*dx
#F2 = Q*q*dx - Q0*q*dx + dt*dot(grad(Q),grad(q))*dx
F = F1+F2

#FF=

Jac = derivative(F, u, du)

# Create nonlinear problem and Newton solver
#problem = NS_self(F, Jac, bcs)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6


ww=2*np.pi/10

PI=3.141592653589793
T=1.1

t = 0.0
Time = 1
t_step=0

Area_data=[]
velocity_data=[]
Inflow_flowrate=[]
Pressure_value=[]
flow_value=[]

cycle=5
total_step=size*cycle

for i in range(0,total_step):
    t += dt
    
    print("time step: "+str(t_step))

    Pin_data=Pin[i-((i//size)*size)][0]
    Pout_data=Pout[i-((i//size)*size)][0]
    Qout_data=Qout[i-((i//size)*size)][0]
    print(Qout_data)
    
    #Inlet area
    Ain=((Pin_data-Pd)*Ad/beta+Ad**(1/2))**2

    #Outlet area
    Aout=((Pout_data-Pd)*Ad/beta+Ad**(1/2))**2

    Vout=Qout_data/Aout

    
    
    #solver.solve(problem, u.vector())
    #solve ( F == 0 , u , bcs, Jac )

    #Inlet Flow Rate set as bc: 
    Qin= (6.5+3.294*np.sin(2*PI*t/T-0.023974)+\
        1.9262*np.sin(4*PI*t/T-1.1801)-\
        1.4219*np.sin(6*PI*t/T+0.92701)-\
        0.66627*np.sin(8*PI*t/T-0.24118)-\
        0.33933*np.sin(10*PI*t/T-0.27471)-\
        0.37914*np.sin(12*PI*t/T-1.0557)+\
        0.22396*np.sin(14*PI*t/T+1.22)+\
        0.1507*np.sin(16*PI*t/T+1.0984)+\
        0.18735*np.sin(18*PI*t/T+0.067483)+\
        0.038625*np.sin(20*PI*t/T+0.22262)+\
        0.012643*np.sin(22*PI*t/T-0.10093)-\
        0.0042453*np.sin(24*PI*t/T-1.1044)-\
        0.012781*np.sin(26*PI*t/T-1.3739)+\
        0.014805*np.sin(28*PI*t/T+1.2797)+\
        0.012249*np.sin(30*PI*t/T+0.80827)+\
        0.0076502*np.sin(32*PI*t/T+0.40757)+\
        0.0030692*np.sin(34*PI*t/T+0.195)-\
        0.0012271*np.sin(36*PI*t/T-1.1371)-\
        0.0042581*np.sin(38*PI*t/T-0.92102)-\
        0.0069785*np.sin(40*PI*t/T-1.2364)+\
        0.0085652*np.sin(42*PI*t/T+1.4539)+\
        0.0081881*np.sin(44*PI*t/T+0.89599)+\
        0.0056549*np.sin(46*PI*t/T+0.17623)+\
        0.0026358*np.sin(48*PI*t/T-1.3003)-\
        0.0050868*np.sin(50*PI*t/T-0.011056)-\
        0.0085829*np.sin(52*PI*t/T-0.86463))
    #Vin=Qin/Ain
    #print("Qin= "+str(Vin))
    print(Ain,Aout,Qin,Qout_data)

    Inflow_flowrate.append(Qin)

    bcA1 = DirichletBC(W.sub(0), Ain, uR_boundary) # BC for Inlet Area
    bcA2 = DirichletBC(W.sub(0), Aout, uL_boundary) # BC for Outlet Area

    bcQ1 = DirichletBC(W.sub(1), Qin, uR_boundary) # BC for Inlet Flowrate
    bcQ2 = DirichletBC(W.sub(1), Qout_data, uL_boundary) # BC for Outlet Flowrate

    bcs = [bcA1,bcA2,bcQ1,bcQ2]



    #A, b=assemble_system(Jac, -F, bcs)

    #solve(A, u.vector(), b)

    #print(u.vector()[:])
    #stop


    #solve(F==0, u , bcs)
    pde = NonlinearVariationalProblem (F , u , bcs , Jac )
    solver = NonlinearVariationalSolver ( pde )
    solver.solve ()





    #print(u.vector()[:])
    #print(np.shape((u.vector()[:])))
    u0.vector()[:] = u.vector()
    
    a1,a2=u.split() 

    test = FunctionSpace(mesh, P1) 

    #test1 = FunctionSpace(mesh, P1)

    a_P1 = project(a1, test) #project area to nodes
    a_nodal_values = a_P1.vector()[:]
    print(a_nodal_values)
    print(np.shape(a_nodal_values))
    
    
    q_P1 = project(a2, test) #project flowrate to nodes
    q_nodal_values = q_P1.vector()[:]
    print(q_nodal_values)
    print(np.shape(q_nodal_values))
    
    coor = mesh.coordinates()

    print('Area:')  # Print area value at each node          
    for i in range(len(a_nodal_values)):
        print (coor[i][0], a_nodal_values[i])

    print('velocity:') # Print flowrate value at each node          
    for i in range(len(q_nodal_values)):
        print (coor[i][0], q_nodal_values[i])

    pressure_data=Pd + beta/Ad * (a_nodal_values**(1/2) - Ad**(1/2))
    #Peak=np.max(pressure_data)
    #pressure_data=pressure_data/Peak
    Flow_rate_data = q_nodal_values#*q_nodal_values
    print("Pressure: "+str(pressure_data))
    print("Flow rate: "+str(Flow_rate_data))
    
    velocity_data.append(q_nodal_values.tolist())
    Area_data.append(a_nodal_values.tolist())
    Pressure_value.append(pressure_data)
    flow_value.append(Flow_rate_data)


    #if t_step >= 10:
    #    stop
    #print(Inflow_velocity)

    t_step += 1

    stop
    
np.savetxt("NS_velocity"+str(Time)+".txt",velocity_data,delimiter=",")    
np.savetxt("NS_area"+str(Time)+".txt",Area_data,delimiter=",")
np.savetxt("Inflow_velocity"+str(Time)+".txt",Inflow_flowrate,delimiter=",") 
np.savetxt("NS_pressure"+str(Time)+".txt",Pressure_value,delimiter=",") 
np.savetxt("NS_flowrate"+str(Time)+".txt",flow_value,delimiter=",") 
