import matplotlib.pyplot as plt
from itertools import product
import numpy as np

#pour la categorical projection, procéder autrement pour ne pas tout recalculer à chaque étape

def wasserstein_metric(d1, d2):
    #get the "cdf"
    x1, y1 = d1.cdf()
    x2, y2 = d2.cdf()
    x1, x2 = [np.NINF] + x1 + [np.inf], [np.NINF] + x2 + [np.inf]
    y1, y2 = [0] + y1, [0] + y2

    #create the diff of CDF
    n1, n2 = len(x1), len(x2)
    i1, i2 = 1, 1
    xd, yd = [], []

    while(i1 < len(x1)-1 or i2 < len(x2)-1):
        if(x1[i1] == x2[i2]):
            xd.append(x1[i1])
            yd.append(abs(y1[i1-1] - y2[i2-1]))
            i1 +=1
            i2 +=1
        elif(x1[i1] < x2[i2]):
            xd.append(x1[i1])
            yd.append(abs(y1[i1-1] - y2[i2-1]))
            i1 += 1
        else:
            xd.append(x2[i2])
            yd.append(abs(y1[i1-1] - y2[i2-1]))
            i2 += 1
    
    #compute the integral of the diff
    res = sum([(xd[i+1] - xd[i])*yd[i+1] for i in range(len(xd)-1)])

    return res

# def tv_distance(mu1:Distribution,mu2:Distribution)->float:
#     atoms = sorted(list(set(mu1._atoms() + mu2._atoms()))

#     return sum([np.abs(mu1_i-mu2_i) for (mu1_i,mu2_i) in zip(mu1._coefficients(),mu2._coefficients())])



def create_distribution_array(dim, dic):
    """creates an array of dimension dim with cells initialized with Distribution()"""
    array = np.ndarray(dim, dtype=object)
    with np.nditer(array, op_flags=["readwrite"], flags=['refs_ok']) as it:
        for cell in it:
            cell[...] = Distribution(dic.copy())
    return array

class Distribution():
    """
        A wrapping of the dic class to represent discretes Distributions for Distributional Reinforcement Learning.
        Implements all the methods et operators to ease the use of those distributions.

    Attributes:
        distrib: the dictionnary representing the discrete distribution
    """

    def __init__(self, dic={0.:1.}):
        """initializes the class with the distribution given as a dictionnary"""
        self.distrib = dic.copy() #essayer de comprendre la copie : https://stackoverflow.com/questions/57342455/how-to-prevent-reassignment-of-variable-to-object-of-same-class-from-keeping-pre

    def project_quantile(self, resolution):
        """
        Implements the quantile regression projection coming from:
        http://arxiv.org/abs/1710.10044

        Args:
            resolution: the number of atoms of the distribution
        """
        new_atoms = []
        new_distrib = {}
        prev_atoms = sorted(list(self._atoms()))
        prev_probas = [self.distrib[prev_atom] for prev_atom in prev_atoms]
        cum_probas = np.cumsum(prev_probas)

        #Compute each quantile
        current_index = 0
        for i in range(resolution):
            while cum_probas[current_index] < (2*i + 1)/(2*resolution):
                current_index += 1
            new_atoms.append(prev_atoms[current_index])

        #creates the new distribution and deals with equal quantiles
        for new_atom in new_atoms:
            if new_atom not in new_distrib.keys():
                new_distrib[new_atom] = 1/resolution
            else:
                new_distrib[new_atom] += 1/resolution 
        
        self.distrib = new_distrib

    def project_categorical(self, resolution, vmin, vmax):
        """
        Implements the categorical regression projection coming from:
        http://arxiv.org/abs/1707.06887

        Args:
            resolution: the number of atoms of the distribution
            Vmin: a lower bound of the distribution
            Vmax: a higher bound of the distribution
        """
        new_atoms = np.linspace(vmin, vmax, resolution)
        new_atoms = np.append(new_atoms, 0) #prevents out of bound indices
        dz = (vmax - vmin)/(resolution - 1)
        new_distrib = {}
        prev_atoms = list(self._atoms())

        for atom in prev_atoms:
            old_proba = self.distrib[atom]
            atom = np.clip(atom, vmin, vmax)
            index = (atom - vmin)/dz

            lower_index = int(np.floor(index))
            upper_index = lower_index + 1
            upper_coef, lower_coef = (index - lower_index), (upper_index - index)
            
            lower_atom, upper_atom = new_atoms[lower_index], new_atoms[upper_index]
            lower_proba, upper_proba = lower_coef*old_proba, upper_coef*old_proba

            if lower_atom not in new_distrib.keys():
                new_distrib[lower_atom] = lower_proba
            else:
                new_distrib[lower_atom] += lower_proba

            if upper_atom not in new_distrib.keys():
                new_distrib[upper_atom] = upper_proba
            else:
                new_distrib[upper_atom] += upper_proba

        self.distrib = new_distrib


    def project_expectile(self, resolution):
        """
        Implements the expectile projection coming from :
        [Thèse de Mastane Achab]

        Args:
            resolution: the number of atoms of the disribution
        """
        new_atoms = []
        new_distrib = {}
        for i in range(resolution):
            new_atoms.append(self.expectile(i/resolution +1e-7,(i+1)/resolution -1e-7))

        #creates the new distribution and deals with equal quantiles
        for new_atom in new_atoms:
            if new_atom not in new_distrib.keys():
                new_distrib[new_atom] = 1/resolution
            else:
                new_distrib[new_atom] += 1/resolution 
        
        self.distrib = new_distrib       

    def normalize(self):
        """
        makes the sum of probabilites equal to 1
        """
        probas = list(self.distrib.values())
        sum_proba = sum(probas)
        if sum_proba == 0:
            self.distrib = dict({0:1})
        for atoms in self._atoms():
            self.distrib[atoms] /= sum_proba

    def transfer(self, gamma, r):
        """returns the affine pushforward transform"""
        items = self.distrib.items()
        items = [(r + gamma*key, item) for (key, item) in items]
        return Distribution(dict(items))

    def convolve(self, other):
        """returns the convolution with another distribution"""
        items1 = self.distrib.items()
        items2 = other.distrib.items()
        sum = dict()

        for ((key1,proba1),(key2,proba2)) in product(items1,items2):
            key = key1 + key2
            if key in sum.keys():
                sum[key] += proba1*proba2
            else:
                sum[key] = proba1*proba2
        return Distribution(sum)

    def rscal(self, scalar):
        """multiplies the random variable by a scalar"""
        items = self.distrib.items()
        items = [(scalar*key, item) for (key, item) in items]
        distrib = dict(items)
        return Distribution(distrib)
    
    def __add__(self,other):
        """sum of measures"""
        distrib = self.distrib.copy()
        for (key,proba) in other.distrib.items():
            if key in distrib.keys():
                distrib[key] += proba
            else:
                distrib[key] = proba
        return Distribution(distrib)

    def __rmul__ (self,scalar):
        """scalar product of a measure"""
        distrib = self.distrib.copy()
        for key in distrib.keys():
            distrib[key] *= scalar
        return Distribution(distrib)

    def __eq__(self, disrib2: object) -> bool:
        """compares two distributions"""
        if not isinstance(disrib2, Distribution):
            return False
        
        atoms1, probas1 = self.cdf() #using .cdf for having it sorted
        atoms2, probas2 = disrib2.cdf()

        return atoms1 == atoms2 and probas1 == probas2
    
    def __getitem__(self, key):
        """Access the distribution directly"""
        if key in self.distrib.keys():
            return self.distrib[key]
        else:
            self.distrib[key] = 0.
            return 0.

    def __setitem__(self, key, item):
        """Modify the distribution directly"""
        self.distrib[key] = item

    def __repr__(self):
        return str(self.distrib)

    def __hash__(self):
        a, p = self.cdf()
        return hash((tuple(a), tuple(p)))

    def _clean(self, threshold=0, rounding=8):
        """Removes from the dictionnary all the atoms with probability lower than threshold"""
        dist = self.distrib.copy()
        for key, item in dist.items():
            if item <= threshold:
                self.distrib.pop(key)
            else:
                self.distrib[key] = round(item, rounding)
    
    def _clean2(self, threshold=1e-5):
        atoms = sorted(self._atoms())
        for i in range(1,len(atoms)):
            if np.abs(atoms[i] - atoms[i-1]) < threshold:
                new_proba = self.distrib[atoms[i]] + self.distrib[atoms[i-1]]
                self.distrib[atoms[i]] = new_proba
                self.distrib[atoms[i-1]] = 0
        self._clean()

    def _atoms(self):
        """Returns the sorted list of atom"""
        return sorted(list(self.distrib.keys()))

    def _values(self):
        """Returns the sorted list of (atom,probability)"""
        atoms = sorted(self._atoms())
        probabilities = [self.distrib[atom] for atom in atoms]
        return zip(atoms,probabilities)

    def _coefficients(self):
        """Returns the *sorted* list of the probability coefficients"""
        atoms = sorted(self._atoms())
        probabilities = [self.distrib[atom] for atom in atoms]
        return probabilities
    
    def _copy(self):
        """returns a copy of the distribution"""
        return Distribution(self.distrib.copy())

    def quantile(self, percent=0.5):
        """return the percent-th quantile"""
        atoms = sorted(self._atoms())
        probabilites = [self.distrib[atom] for atom in atoms]
        cum_sum = np.cumsum(probabilites)
        
        i=0
        while(cum_sum[i] < percent):
            i+=1
        
        return atoms[i]

    def expectile(self, low, up):
        """return the percent-th expectile"""
        #To clean
        atoms = sorted(self._atoms())
        probabilites = [self.distrib[atom] for atom in atoms]
        cum_sum = np.cumsum(probabilites)
        
        #find the atom that reachs the low-th quantile
        res = 0
        i=0 
        while(cum_sum[i] <= low):
            i+=1

        #if it fills low and up, returns that atom
        if cum_sum[i] >= up:
            return atoms[i]

        #add the participation of the lower quantile
        res += atoms[i]*(cum_sum[i] - low)
        i+=1

        #add the quantiles fully covered by the span of low-up
        while cum_sum[i] < up:
            res += atoms[i]*probabilites[i]
            i+=1

        #add the participation of the upper quantile
        res += atoms[i]*(up-cum_sum[i-1])

        return res/(up-low) 

    def exp_utility(self, beta):
        return np.sign(beta)*sum([proba * np.exp(beta*atom) for (atom,proba) in self._values()])

    def entRM(self, beta):
        if beta == 0:
            return self.mean
        else:
            return np.log(sum([mu_i*np.exp(beta*x_i) for (x_i,mu_i) in self._values()]))/beta
    
    def cdf(self):
        atoms = sorted(self._atoms())
        probabilities = list(np.cumsum([self.distrib[atom] for atom in atoms]))
        return atoms,probabilities
    
    def support(self):
        atoms = sorted(self._atoms())
        return [atoms[0], atoms[-1]]

    @property
    def mean(self):
        return sum([atom*proba for (atom,proba) in self._values()])

    def plot(self, title=None, figsize=(15, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.margins(x=0)
        x = self.distrib.keys()
        y = self.distrib.values()
        ax.bar(x,y)
        ax.set_title(title)
        ax.set_xlabel("Atoms")
        ax.set_ylabel("Probability")

############################################################################################################
        
class DistributionArray():
    """
        Implements an array of Distribution with methods that gets apply to every cell.
    
        Args:
            dim: dimension of the array
            dic: dictionnary to initialize each distribution with
    """
    def __init__(self, dim, dic={0.:1.}):
        self.dim = dim
        self.array = create_distribution_array(dim, dic)

    def project_quantile(self, resolution):
        for idx in np.ndindex(self.dim):
            self.array[idx].project_quantile(resolution)

    def project_categorical(self, resolution, vmin, vmax):
        for idx in np.ndindex(self.dim):
            self.array[idx].project_categorical(resolution, vmin, vmax)
    
    def project_expectile(self, resolution):
        for idx in np.ndindex(self.dim):
            self.array[idx].project_expectile(resolution)

    def normalize(self):
        for idx in np.ndindex(self.dim):
            self.array[idx].normalize()

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def clean(self, threshold):
        for idx in np.ndindex(self.dim):
            self.array[idx]._clean(threshold)

    def copy(self):
        new_dspace = np.ndarray(self.dim,dtype=object)
        for idx, distrib in np.ndenumerate(self.array):
            new_dspace[idx] = distrib._copy()
        return new_dspace


