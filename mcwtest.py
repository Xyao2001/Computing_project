import numpy as np
import pandas as pd
import random
import matplotlib as plt
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
    initial_state = np.array([1,0]) 
    global H_eff

    # H_eff=np.array([[delta,omega],[omega,-1*delta-(1j*gamma)]])
    H_eff=np.array([[0,0.06],[0.06,-(1j*0.03)]])

    #random.random() function to generate a random floating-point number between 0 and 1 in Python.
    def __init__(self,delta,omega,gamma):
      
        #delta=laser_frequency-qubit_resonant_frequency
        # omega is known as the Rabi frequency and is proportional to the laser field amplitude
        #gamma is the decay rate
        
       # H_eff=np.array([[delta,omega],[omega,-delta-1j*gamma]])
       # return H_eff
       None

    

    

    def evolve(self,shell_number,time):
        time=time
        n=shell_number
        dt=1
        sum_wave_vector=[]
        '''
        # evolve the wavevector by a very small time step
        evolved_=(1-((1j)*(H_eff*dt)))*initial_state
        #Then, choose a random number r between 0 and and 1
        r=random.randint(0,1)
        print(f'r is %%%%%%%%{r}')
        evolved_star=evolved_.conj().T
        norm_=(np.sum(np.multiply(evolved_star,evolved_))).real
        norm_squared=int(norm_)
        '''
        sum_time=dt
        psi=initial_state-1-(sum_time/2j*(H_eff*initial_state))
        psi/=np.sum(np.square(np.abs(psi)))
        #normalise psi
        
        #print(f"effective hermitian is {H_eff}  initial state is {initial_state}, h_eff on initial State is {H_eff*initial_state}")
        while sum_time!=time: 
            sum_time+=dt
            #coefficient=np.sum(1-np.sum((1j*(H_eff*sum_time))))
            coefficient=(1-((1j*(H_eff*sum_time))))
            evolved_=coefficient*psi
            norm_squared=np.sum(np.square(np.abs(evolved_)))
            
            #Then, choose a random number r between 0 and and 1
            r=random.uniform(0,1)
            #print(f"r is {r}, and norm squared is {norm_squared}")
            
            #If r is less than the norm-squared h then there is no decay and we simply renormalise
            if (r<norm_squared):

                # then there is no decay and we simply renormalise
                psi=evolved_/np.sqrt(norm_squared)
                #print(f"no jump and wave vector is {psi}")
                sum_wave_vector.append(psi)
                 
        #if r is greater than or equal to norm _squared then there's a decay
            elif r>=norm_squared:
            
                #we apply a jump operator
                gamma=0.03
                #print('time evo operator is being applied')
                #je_star is the hermitian of time_evo, i know these names are awful
                jump_operator=math.sqrt(gamma/2)*np.array([[0,1],[0,0]])
                psi_1=jump_operator*evolved_
                #print(f" ::::psi is {evolved_}:the jump operator acting on psi is:::::: {psi_1}")
                #the wave vector is giving nan values
                
                norm_square=np.sum(np.square(np.abs(psi_1)))
                
                psi=psi_1/np.sqrt(norm_square)
                sum_wave_vector.append(psi)
        sum_wave_vector=(sum_wave_vector)
        for i in range(len(sum_wave_vector)):
            #print the excited populations
            arr=(sum_wave_vector[i])
            print(f"the excited probability is {arr[1,1].real}")
   
        return sum_wave_vector
        
    

    def transpose_conjugate(self, n, matrix=np.ndarray ):
        density_matrix_sum=[]

        for i in range(n):
            density_matrix_sum.append(matrix[i].T.conj())

            return density_matrix_sum

    def MC_Sampling(self, n_trajectory,time):
        num=n_trajectory
        r=random.random()
        excitations=[]
        sum_traj=self.evolve(2,time)
        
        sum_t_c=self.transpose_conjugate(num, sum_traj)
        density_matrix=(1/num)*sum_t_c
        return density_matrix[3]
        #

time=10
times=np.arange(time)

Monte=MC(0,1,2)
print(Monte.evolve(2,3))

arr=np.array([1,0])