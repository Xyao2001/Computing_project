import numpy as np

#create 2D numpy array with zeros
a = np.array([[1-1j,0],[1-1j,0]])
a_star=np.conjugate(np.transpose(a))
norm_squared=np.sum(np.square(np.abs(a)))
  


#print numpy array
print(f"   {a}    \n   {a_star} \n norm squared is {norm_squared}")