from numpy import *
import matplotlib.pyplot as plt
from copy import deepcopy
from operator import itemgetter
from collections import Counter
import scipy as scipy
import random
from scipy.spatial.distance import pdist, squareform
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as multiprocessing
from aux_funcs import *

# this is the main function that runs everything.
def main():

  list_elm=rd_data()                       ##  read the data
  t,h,ho=create_sets(list_elm)             ##  create training, hold_out and test sets. There is some flexibility in the creation. Read function for further details
  A1=KRR(t,h,ho,'gaussian')                           ##  creates the KRR class for the specified  training/hold/test sets

#  for lbda in [2**(-32)]:
##    for sigma in [2**10]:
#  for lbda in 2**(arange(-40,-5.5,0.5)):
#    for sigma in 2**arange(5,18.5,0.5):
  error_map=[]
  lbda_vals=arange(-40,-4.5,1.5)
  sigma_vals=arange(5,18.5,1.5)
  for lbda in 2**lbda_vals:
    for sigma in 2**sigma_vals:
#      print lbda,sigma
      A1.training(lbda,sigma)
      A1.predict('hold_out')
      A1.error_estimate()
      error_map.append(A1.RMSE)

  error_map=array(error_map);error_map=error_map.reshape(len(lbda_vals),len(sigma_vals))
#  A1.predict('test')
  lbda,sigma=meshgrid(arange(-40,-4.5,1.5),arange(5,18.5,1.5))
  plt.contourf(sigma.T,lbda.T,error_map,50); plt.colorbar()
  return 0 #error_map,A1


def rd_data():
  with open('../resources/dsgdb7ae2.xyz') as f:
    content = f.readlines()

  j=0
  list_elm=[]
  id=1
# Loop for reading everything and making a list of molecule Class C
  while j <len(content):
    natoms=int(content[j]); j+=2; 
    new_elm=C(natoms,content[j-1])   # create class C
    new_elm.build_qxyz(content[j:j+natoms])
    list_elm.append(new_elm)
    j=j+natoms
    id=id+1

  list_elm.sort( key=lambda x: x.n_nonH, reverse=False)  #reorder list with respect to # of nH
  list_elm=array(list_elm);

  return list_elm



class KRR():

  def __init__(self,training_set,hold_out_set,test_set,kernel_type):

  ## define training set
    self.kernel_type=kernel_type
    self.training_C_vector  =[]
    self.training_energy    =[]
    self.training_id=[]
    self.training_dim=len(training_set)
    for j in arange(0,self.training_dim):
      self.training_energy.append(training_set[j].energy)
      self.training_C_vector.append(training_set[j].C_vector)
      self.training_id.append(training_set[j].id) 


    self.training_C_vector  =  array(self.training_C_vector); self.training_C_vector=self.training_C_vector.reshape(self.training_dim,276)
    self.training_energy    =  array(self.training_energy)

  ## define hold_out set
    self.hold_out_C_vector  =[]
    self.hold_out_energy    =[]
    self.hold_out_dim=len(hold_out_set)
    self.hold_out_id=[]

    for j in arange(0,self.hold_out_dim):
      self.hold_out_energy.append(hold_out_set[j].energy)
      self.hold_out_C_vector.append(hold_out_set[j].C_vector)
      self.hold_out_id.append(hold_out_set[j].id)
    self.hold_out_C_vector  =  array(self.hold_out_C_vector); self.hold_out_C_vector=self.hold_out_C_vector.reshape(self.hold_out_dim,276)
    self.hold_out_energy    =  array(self.hold_out_energy)

  ## define test set
    self.test_C_vector  =[]
    self.test_energy    =[]
    self.test_dim=len(test_set)
    self.test_id=[]

    for j in arange(0,self.test_dim):
      self.test_energy.append(test_set[j].energy)
      self.test_C_vector.append(test_set[j].C_vector)
      self.test_id.append(test_set[j].id)
    self.test_C_vector  =  array(self.test_C_vector); self.test_C_vector=self.test_C_vector.reshape(self.test_dim,276)
    self.test_energy    =  array(self.test_energy)

  def error_estimate(self):
    self.RMSE=rmse(self.pred,self.hold_out_energy)
    return
  def training(self,lbda,sigma):
    # this instance trains the data
    self.lbda=lbda
    self.sigma=sigma

    #pairwise_dists is a very smart way of calculating the difference between all C_vectors of the training set.
#    pairwise_dists = squareform(pdist(self.training_C_vector, 'minkowski',1))
    
    K = Kernel(self.sigma,self.kernel_type,self.training_C_vector,'training')
#    K=    scipy.exp(-pairwise_dists  / self.sigma /1.)   # training Kernel Matrix
    U=transpose(linalg.cholesky(K+self.lbda*identity(self.training_dim)))  #U is upper triangular. K=K+lambda*I  for including regularization
    self.K=K
    self.U=U
    self.alpha=Rasmussen(U,Rasmussen(transpose(U),self.training_energy)) # all we need in the end from this function is alpha.
#    return self.training_C_vector[3],self.training_C_vector[7],K[3,7]


  def predict(self,predict_set):
    #This instance predicts the energies of the hold_out or test sets 

    if predict_set=='hold_out':
      predict_array=self.hold_out_C_vector
      dim=self.hold_out_dim
    if predict_set=='test':
      predict_array=self.test_C_vector
      dim=self.test_dim

    # Stupid way of calculating the prediction K

    Kpred=zeros(dim*self.training_dim).reshape(self.training_dim,dim)
    for j in arange(0,dim):
      Kpred[:,j] = Kernel(self.sigma,self.kernel_type,self.training_C_vector,array(predict_array[j,:]),'predict')
    Kpred=transpose(Kpred)  
#    print shape(Kpred),shape(self.alpha)
    self.pred=Kpred.dot(self.alpha)
    return 0#Kpred.dot(self.alpha)


class C:

  def __init__(self,natoms,energy):
    self.natoms=natoms
    self.id=float(energy.split()[0])
    self.list_atoms=[]
    self.energy=float(energy.split()[1])
    self.nH=0
    self.n_nonH=0
    self.C_vector=[]
    self.C_matrix=zeros(23*23).reshape(23,23)

  def build_qxyz(self,M):

    self.q_xyz=zeros(self.natoms*4).reshape(self.natoms,4)
    for j in arange(0,self.natoms):
      self.q_xyz[j,0]=check_atom(M[j][0])
      self.list_atoms.append(M[j][0])
      get_pos=str.split(M[j][5:43])

      self.q_xyz[j,0]=check_atom(M[j][0])
      self.q_xyz[j,1]=float(get_pos[0])
      self.q_xyz[j,2]=float(get_pos[1])
      self.q_xyz[j,3]=float(get_pos[2])
    self.build_C_matrix()
    self.nH=self.list_atoms.count('H')
    self.n_nonH=self.natoms-self.nH

  def build_C_matrix(self):
  
    for a in arange(0,self.natoms):
      for b in arange(0,self.natoms):

        charge_charge=self.q_xyz[a,0]*self.q_xyz[b,0]
        if a!= b:
          dist=sum((self.q_xyz[a,1:4]-self.q_xyz[b,1:4])**2)*1.889725989**2  #1.88 to convert from angstrom to bohr
          self.C_matrix[a,b]=charge_charge/dist
        else:
          self.C_matrix[a,b]=0.5*charge_charge**(2.4/2.) 
# bit to reorder the C Matrix
    self.C_matrix=sort_matrix(self.C_matrix)
    for j in arange(0,23):
      self.C_vector+=array(self.C_matrix[0:(j+1),j]).tolist() 
#    zero_pads=276-self.natoms**2

  def check_n_nH_atoms():
    self.nH=self.list_atoms.count('H')
    self.n_nonH=self.natoms-self.nH

     

def Kernel(sigma,kernel_type,v1,v2=array([0]),flag='training'):
  if flag=='training':   # this is the training set
    if kernel_type == "gaussian" :
      pairwise_dists = squareform(pdist(v1, 'euclidean'))
#      print amax(pairwise_dists**2),amin(pairwise_dists**2),sigma**2

      return scipy.exp(-pairwise_dists ** 2 / sigma ** 2 /2.)   # training Kernel Matrix

  if flag=='training':   # this is the training set
    if kernel_type == "euclidean_p_sigma" :
      pairwise_dists = squareform(pdist(v1, 'euclidean'))
      print amax(pairwise_dists**2),amin(pairwise_dists**2),sigma**2
      return 1-pairwise_dists ** 2 / sigma ** 2 /2. +0.5*(pairwise_dists ** 2 / sigma ** 2 /2.)**2  # training Kernel Matrix


    if kernel_type=='laplacian':
      pairwise_dists = squareform(pdist(v1, 'minkowski',1))
      return scipy.exp(-pairwise_dists  / sigma /1.)   # training Kernel Matrix


  dists=abs(v1-v2)
  if kernel_type == "gaussian" :
    return scipy.exp(-sum(dists ** 2,axis=1) / sigma ** 2 /2.)   # training Kernel Matrix

  if kernel_type == "euclidean_p_sigma" :
    return 1-dists ** 2 / sigma ** 2 /2.++0.5*(pairwise_dists ** 2 / sigma ** 2 /2.)**2   # training Kernel Matrix


  if kernel_type=='laplacian':
    return scipy.exp(-sum(dists,axis=1)  / sigma /1.)   # training Kernel Matrix


main()
