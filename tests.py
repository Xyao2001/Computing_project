import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import seaborn as sea
class OBE:
    def init(self):
        None

    def analytical_solution(self, time,gamma, omega):

        results=[]
        for i in range(time):
            t=i
            expo_term=(math.exp(-(3gammat)/4))
            term_1=(math.cos(math.sqrt(omega2-((gamma2)/16))t))
            term_2=((3gamma)/(math.sqrt(16(omega2)-(gamma2))))math.sin(math.sqrt((omega2)-(gamma2)/16)t)
            p_11=(omega**2/(2(omega2)+(gamma2)))(1-(expo_term(term_1+term_2)))
            results.append([p_11])

        return results

obe=OBE()
time=1000
x=np.arange(time)
omega=0.1
y=obe.analytical_solution(time,0.1*omega,omega)

plt.figure()
plt.plot(x,y)


plt.show()
