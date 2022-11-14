import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import seaborn as sea
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
x=np.arange(time)

y=obe.analytical_solution(time,0.03,0.06)

plt.figure()
plt.plot(x,y)


plt.show()
