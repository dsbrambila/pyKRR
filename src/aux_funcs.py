from copy import deepcopy
from numpy import *
import random
def Rasmussen(U,y):

  dim=len(U)
  print dim
  beta=zeros(dim)
  if U[0,1]==0 :   # Lower triangular matrix
    for i in arange(0,dim):
      beta[i]=(  y[i]-sum(U[i,0:i]*beta[0:(i)])  )/U[i,i]
    return beta
  else:
    for i in arange(dim-1,-1,-1):
      beta[i]=(  y[i]-sum(U[i,dim:(i):-1]*beta[dim:(i):-1])  )/U[i,i]
    return beta

def create_sets(list_elm,nh_less_5=58,size_ho=100,size_tpho=1000,rand_number=sort(random.sample(range(100, 999), 100))):
  # This function creates the training, holdout and test sets
  #There are 58 molecules with less than 5 hydrogens. Thats why they are all included in the training+holdout set
  #       The other molecules included in the training+hold_out set are controlled by the rand_number variables.
  #size_tpho determines the size of the training+hold out sets. 

  if size(rand_number) != size_ho:
    print 'len(rand_number)!= size_ho !!!!', size_ho,size(rand_number)
    return 0

  #This part excludes the training set from the test set
  index_tset=arange(0,nh_less_5).tolist()+linspace(nh_less_5,7101,size_tpho-nh_less_5).astype(int).tolist()
  tph_set=list_elm[array(index_tset)] #create training+holdout set
  test_set=list_elm[setxor1d(arange(0,7101),array(index_tset))] #create test_set


  #Separating training+ holdout set
  hold_out_set=tph_set[rand_number]                         ##create hold_out set 
  training_set=tph_set[setxor1d(arange(0,size_tpho),rand_number)]##create training set complimentary to holdout

  return training_set,hold_out_set,test_set #I return everything as ndarrays



def rmse(predictions, targets):
    return sqrt(((array(predictions) - array(targets)) ** 2).mean())

def check_atom(atom):
  if atom=='H': return 1.0
  if atom=='C': return 6.0
  if atom=='O': return 8.0
  if atom=='N': return 7.0
  if atom=='S': return 16.0
def sort_matrix(M):

# first order elemens of the rows
## Example:

##   A=   2  1   3   1  5   norm= 12
##        1  8   2   6  1   norm= 18
##        3  2   2   6  1   norm= 14
##        1  6   6   7  6   norm= 26 
##        5  1   1   6  4   norm= 17
# Permutation 4X1
##   A=   7  6   6   1  6   norm= 26
##        6  8   2   1  1   norm= 18
##        6  2   2   3  1   norm= 14
##        1  1   3   2  5   norm= 12 
##        6  1   1   5  4   norm= 17
# Permutation 2X2....
# Permutation 5X3
##   A=   7  6   6   1  6   norm= 26
##        6  8   1   1  2   norm= 18
##        6  1   4   5  1   norm= 17
##        1  1   5   2  3   norm= 12 
##        6  2   1   3  2   norm= 14

# Permutation 4X5
##   A=   7  6   6   6  1   norm= 26
##        6  8   1   2  1   norm= 18
##        6  1   4   1  5   norm= 17
##        6  2   1   2  3   norm= 14
##        1  1   5   3  2   norm= 12 

#make set with norm values
  norm_set=sum(abs(M)**2,axis=-1)**(1./2)         #  list of norms
  for perm_min in arange(0,23):
    perm_max=argmax(norm_set[perm_min::])+perm_min
    aux=deepcopy(max(norm_set[perm_min::]))
    norm_set[perm_max]=norm_set[perm_min]
    norm_set[perm_min]=aux
    if perm_min!=perm_max:
      aux=deepcopy(M[perm_min,:])
      M[perm_min,:]=M[perm_max,:]
      M[perm_max,:]=aux
      aux=deepcopy(M[:,perm_min])
      M[:,perm_min]=M[:,perm_max]
      M[:,perm_max]=aux
  return M

