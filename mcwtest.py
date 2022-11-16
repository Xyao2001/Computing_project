# Online Python compiler (interpreter) to run Python online.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math



class MC:
    global pho_matrix
    global r
    global gamma
    #quantum number
    n=1
    global h_bar
    h_bar=1
    global initial_state
    initial_state = np.array([[1],[0]]) 
    global H_eff

    # H_eff=np.array([[delta,omega],[omega,-1*delta-(1j*gamma)]])
    H_eff=np.array([[0,0.1],[0.1,-(1j*0.01)]])

    #random.random() function to generate a random floating-point number between 0 and 1 in Python.
    def __init__(self,delta,omega,gamma):

        #delta=laser_frequency-qubit_resonant_frequency
        # omega is known as the Rabi frequency and is proportional to the laser field amplitude
        #gamma is the decay rate

       # H_eff=np.array([[delta,omega],[omega,-delta-1j*gamma]])
       # return H_eff
       None





    def evolve(self,shell_number,time,steps):
        time=time
        n=shell_number
        dt=steps
        m_1=np.array([[1,0],[0,1]])
        excited_states=[]
        sum_wave_vector=[]
        H_eff=np.array([[0,0.1],[0.1,-(1j*0.01)]])
        sum_time=0
        psi=initial_state
        #print(f"effective hermitian is {H_eff}  initial state is {initial_state}, h_eff on initial State is {H_eff*initial_state}")
        while sum_time<time: 
            sum_time+=dt
            
            #coefficient=np.sum(1-np.sum((1j*(H_eff*sum_time))))
            coefficient=m_1-(0.5j)*H_eff
            psi=np.matmul(coefficient,psi)
            norm=np.linalg.norm((psi))
            
            #norm_squared=(np.abs(np.matmul(np.conj(np.transpose(evolved_)),evolved_)))[1,1]
            norm_squared=norm*norm
            r=random.random()
            #If r is less than the norm-squared h then there is no decay and we simply renormalise
            if (r<norm_squared):

                # then there is no decay and we simply renormalise
                psi=psi/np.sqrt(norm_squared)
                #print(f"no jump and wave vector is {psi}")
                sum_wave_vector.append(abs(psi[1]**2))

        #if r is greater than or equal to norm _squared then there's a decay
            elif r>=norm_squared:

                #we apply a jump operator
                gamma=0.01
                #print('time evo operator is being applied')
                #je_star is the hermitian of time_evo, i know these names are awful
                jump_operator=math.sqrt(gamma/2)*np.array([[0,1],[0,0]])
                psi_1=np.matmul(jump_operator,psi)
                #print(f" ::::psi is {evolved_}:the jump operator acting on psi is:::::: {psi_1}")
                #the wave vector is giving nan values

                norm_square=np.linalg.norm((psi_1))*np.linalg.norm((psi_1))

                psi=psi_1/np.sqrt(norm_square)
                sum_wave_vector.append(abs(psi[1]**2))
            
        

        return sum_wave_vector



    def transpose_conjugate(self, n, matrix=np.ndarray ):
        density_matrix_sum=[]

        for i in range(n):
            density_matrix_sum.append(matrix[i].T.conj())

            return density_matrix_sum

class OBE:
    def __init__(self):
        None

    def analytical_solution(self, time,gamma, omega):
        
        results=[]
        for i in range(time):
            t=i
            expo_term=(math.exp(-(3*gamma*t)/4))
            term_1=(math.cos(math.sqrt(omega**2-((gamma**2)/16))*t))
            term_2=((3*gamma)/(math.sqrt(16*(omega**2)-(gamma**2))))*math.sin(math.sqrt((omega**2)-(gamma**2)/16)*t)
            p_11=(omega**2/(2*(omega**2)+(gamma**2)))*(1-(expo_term*(term_1+term_2)))
            results.append([p_11])
        
        return results

obe=OBE()

time=1000
steps=1
times=np.arange(start=0, stop=time, step=steps)

Monte=MC(0,1,2)



y=(Monte.evolve(2,time,steps))

avg=[]

for i in range(time):
    monte=Monte.evolve(2,time,steps)
    avg.append(monte) #list of lists
arr=np.array(avg)
y1 = np.average(arr, axis=0)

y2=obe.analytical_solution(time,0.01,0.1)
plt.figure()
plt.plot(times,y, label = "single MCW")
plt.plot(times,y1, label = "Averaged MCW")
plt.plot(times,y2, label = "OBE")
plt.legend()

plt.show()