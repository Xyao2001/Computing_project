# Online Python compiler (interpreter) to run Python online.
import numpy as np
import random
import matplotlib.pyplot as plt
import math
class MC:
    #quantum number
    n=1
    global initial_state
    initial_state = np.array([[1],[0]]) 

    def __init__(self,delta,omega,gamma):

       None





    def evolve(self,shell_number,time,steps):
        time=time
        n=shell_number
        dt=steps
        m_1=np.array([[1,0],[0,1]])
        excited_states=[]
        sum_wave_vector=[]
        #*dt
        #H_eff=np.array([[0,0.1*dt],[0.1*dt,-(1j*0.01*dt)]])
        sum_time=0
        psi=initial_state
        #print(f"effective hermitian is {H_eff}  initial state is {initial_state}, h_eff on initial State is {H_eff*initial_state}")
        while sum_time<time: 
            sum_time+=1
            coefficient=np.array([[1,-0.5j*dt*0.1],[-0.5j*dt*0.1,1-0.01*dt/2]])
            psi=np.matmul(coefficient,psi)
            norm=np.sqrt(abs(psi[0])**2+abs(psi[1])**2)
            norm_squared=norm*norm
            r=random.random()
            #If r is less than the norm-squared h then there is no decay and we simply renormalise
            if (r<norm_squared):

    
                psi=psi/np.sqrt(norm_squared)
                #print(f"no jump and wave vector is {psi}")
                sum_wave_vector.append(abs(psi[1]**2))

        #if r is greater than or equal to norm _squared then there's a decay
            elif r>=norm_squared:

                #gamma=0.01
                jump_operator=math.sqrt(0.01/2)*np.array([[0,1],[0,0]])
                psi_1=np.matmul(jump_operator,psi)
                norm_1=np.sqrt(abs(psi_1[0])**2+abs(psi_1[1])**2)
                norm_square=norm_1*norm_1

                psi=psi_1/np.sqrt(norm_square)
                sum_wave_vector.append(abs(psi[1]**2))
            
        

        return sum_wave_vector

time=5000
ntime=5000
steps=0.1
dt=steps

Ga=0.01
Om=0.1

A = 2*Om**2/(2*Om**2+Ga**2)
B = 3*Ga/(np.sqrt(16*Om**2-Ga**2))
C = np.sqrt(Om**2-Ga**2/16)
def OBE(t):
    return A*(1-np.exp(-3*Ga*t/4)*(np.cos(C*t)+B*np.sin(C*t)))
y_OBE = []
for i in range(0,ntime):
    y_OBE += [0.5*OBE(i*dt)]

y2 = np.array(y_OBE)
print(f"shape obe is {y2.shape}")

Monte=MC(0,1,2)
y=(Monte.evolve(2,time,steps))
avg=[]
for i in range(ntime):
    monte=Monte.evolve(2,time,steps)
    avg.append(monte) #list of lists
arr=np.array(avg)
y1 = (np.average(arr, axis=0)).flatten()
residuals=y2-y1


plt.figure()

import matplotlib.ticker as ticker  # Allows you to modify the tick markers to

xtick_spacing = 500  # assess the errors from the chi-squared
ytick_spacing = 0.2  # contour plots.

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

times=np.arange(start=0, stop=time, step=1)
#plt.plot(times,y, label = "single MCW")
plt.plot(times,y1, label = "Averaged MCW", color="gold")
plt.plot(times,y2, label = "OBE", linestyle="dashed", color="deepskyblue")
plt.plot(times,residuals, label = "Residuals", linestyle="dotted", color="hotpink")
plt.legend()
plt.xlabel('Time (1/100 Rabi period)')  # Axis labels
plt.ylabel('Excited state population')


plt.show()