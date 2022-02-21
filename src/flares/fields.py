from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from flares.utils import *
from flares.data import *

import torch
import torch.nn.functional as F
import numpy as np

import time


mu0 = 4 * np.pi * 10**-3
k = 1e3
M = 1e6
ds = np.sqrt(1.33e105) * k / M


class ActiveRegionFields:

    def __init__(self, Bz, By, Bx):
        """A class to write "derived" fields from x, y, z components of magnetic field (and possibly more)
        This class is designed so that derrived fields are only computed once, and 
        operates on an "assertion leve", so that if a field already exists, nothing 
        is done

        Args:
            Bz (2d np.array): The z component magnetic flux
            By (2d np.array): The y component magnetic flux
            Bx (2d np.array): The x component magnetic flux
        """
        assert Bz.shape == By.shape and Bz.shape == Bx.shape

        self.Bz = Bz
        self.By = By
        self.Bx = Bx

        self.shape = Bz.shape

    def manually_add_field(self, field, field_name):
        """If a desired field doesn't exist or is arbitrary, use this function. Be careful and make sure
        the field you use is referencing the same active region

        Args:
            field ([type]): a 2d array the same shape as self.bz
            field_name ([type]): [description]
        """
        assert field.shape == self.shape

        if not hasattr(self, field_name):
            setattr(self, field_name, field)

    def assert_Bh(self):
        """Horizontal Magnetic Field Component

        $$B_h = norm(B_x, B_y)$$
        """
        if not hasattr(self, 'Bh'):
            self.Bh = self.norm((self.Bx, self.By))
    def assert_gamma(self):
        """Angle of line of magnetic flux vector from the horizontal

        $$\\gamma = arctan(\\frac{B_z}{|B_x| + |B_y|})$$
        """
        if not hasattr(self, 'gamma'):
            self.assert_Bh()
            self.gamma = np.arctan(self.Bz / self.Bh)
    def assert_B(self):
        """Magnitude of magnetic flux vector

        $$B = norm(B_x, B_y, B_z)$$
        """
        if not hasattr(self, 'B'):
            self.B = self.norm((self.Bx, self.By, self.Bz))
    def assert_grad_B(self):
        """Gradient of magnetic field magnitude

        $$B = norm(B_x, B_y, B_z)$$
        """
        if not hasattr(self, 'grad_B_x') or not hasattr(self, 'grad_B_y'):
            self.assert_B()
            self.grad_B_x, self.grad_B_y = gradient(self.B)
    def assert_grad_Bh(self):
        """Gradient of horizontal magnetic field

        $$B_h = norm(B_x, B_y)$$
        """
        if not hasattr(self, 'grad_Bh_x') or not hasattr(self, 'grad_Bh_y'):
            self.assert_Bh()
            self.grad_Bh_x, self.grad_Bh_y = gradient(self.Bh)
    def assert_grad_Bz(self):
        """Gradient of line of site magnetic field
        """
        if not hasattr(self, 'grad_Bz_x') or not hasattr(self, 'grad_Bz_y'):
            self.grad_Bz_x, self.grad_Bz_y = gradient(self.Bz)
    def assert_grad_Bx(self):
        """Gradient of line of x component of magnetic field
        """
        if not hasattr(self, 'grad_Bx_x') or not hasattr(self, 'grad_Bx_y'):
            self.grad_Bx_x, self.grad_Bx_y = gradient(self.Bx)
            self.grad_Bx_x = -self.grad_Bx_x
    def assert_grad_By(self):
        """Gradient of y component of magnetic field
        """
        if not hasattr(self, 'grad_By_x') or not hasattr(self, 'grad_By_y'):
            self.grad_By_x, self.grad_By_y = gradient(self.By)
            self.grad_By_x = -self.grad_By_x
    def assert_grad_Bm(self):
        """Magnitude of gradient vectors of x, y, z magnetic fields

        $$\\nabla B = norm(\\nabla B_z, \\nabla B_y, \\nabla B_z)$$
        """
        if not hasattr(self, 'grad_Bm'):
            self.assert_grad_B()
            self.grad_Bm = self.norm((self.grad_B_x, self.grad_B_y))
    def assert_J(self):
        """Vertical current density

        $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$
        """
        if not hasattr(self, 'J'):
            self.assert_grad_Bx()
            self.assert_grad_By()
            self.J = (self.grad_By_x - self.grad_Bx_y) / mu0
            self.J[self.Bz == 0] = 0
    def assert_Jh(self):
        """Vertical heterogeneity current density

        $$J_z^h = \\frac{1}{B}(B_y\\frac{\\partial B_x}{\\partial y} - B_x \\frac{\\partial B_y}{\\partial x})$$
        """
        if not hasattr(self, 'Jh'):
            self.assert_grad_Bx()
            self.assert_grad_By()
            self.assert_B()
            self.Jh = (self.By * self.grad_Bx_y - self.Bx * self.grad_By_x) / mu0
    def assert_hc(self):
        """Current helicity

        $$h_c = B_z(\\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y})$$
        """
        if not hasattr(self, 'hc'):
            self.assert_J()
            self.hc = self.Bz * self.J
    def assert_shear(self):
        """Shear angle

        $$\\Psi = arccos(\\frac{\\vec{B_p}\\cdot\\vec{B_o}}{|B_o||B_p|})$$
        """
        if not hasattr(self, 'shear'):
            self.assert_Bp()
            self.assert_B()
            dot = self.Bx * self.Bpx + self.By * self.Bpy + self.Bz * self.Bpz
            magp = self.norm((self.Bpx, self.Bpy, self.Bpz))
            self.shear = np.arccos(dot / (self.B * magp))
    def assert_rho(self):
        """Excess magnetic energy density

        $$\\rho_e = norm(B_p - B_o)$$
        """
        if not hasattr(self, 'rho'):
            self.assert_Bp()
            self.rho = (self.B - self.norm((self.Bpx, self.Bpy, self.Bpz)))**2 / (8*np.pi)

    def assert_Bp(self):
        """Magnetic field vector assuming zero current

        $$B = -\\nabla \\phi \\quad \\nabla^2 \\phi= 0 \\quad (z > 0)$$
        $$-\\hat{n}\\cdot\\nabla\\phi = B_n \\quad (z = 0)$$
        $$\\phi(r) \\rightarrow 0 \\quad as \\quad r \\rightarrow \\infty \\quad (z = 0)$$

        Use green's function for the neumann boundary condition
        $$\\nabla^2 G_n(r, r') = 0 \\quad (z > 0)$$
        $$G_n \\rightarrow 0 \\quad as \\quad |r - r'| \\rightarrow \\infty \\quad (z > 0)$$
        $$-\\hat{n}\\cdot\\nabla G_n = 0 \\quad (z = 0, r' \\neq r)$$

        $$-\\hat{n}\\cdot\\nabla G_n$$ 
        
        diverges to keep unit flux

        $$lim_{z \\rightarrow 0^+}\\int\\hat{n}\\cdot\\nabla G_n(r,r')dS = 1$$
        
        We have the solution:

        $$\\phi(r) = \\int B_n(r')G_n(r, r')dS' \\quad dS' = dx'dy' \\quad r' = (x', y', 0)$$

        Explicit form:
        $$G_n(r, r') = \\frac{1}{2\\phi R} \\quad (R = |r - r'|)$$
        
        Discrete form:
        $$\\phi(r) \\rightarrow \\sum_{r'_{ij}}B_n(r')\\tilde{G}_n(r, r'_{ij})\\Delta^2$$
        
        and $$B_n$$ is approximated by:

        $$B_n(r) \\rightarrow -\\sum_{r'_{ij}}\\hat{n}\\cdot\\nabla \\tilde{G}_n(r, r'_{ij})\\Delta^2 \\quad (r = (x, y, 0))$$
        
        $$\\tilde{G}_n(r, r'_{ij}) = \\frac{1}{2\\pi |r - r'_{ij} + (\\Delta/\\sqrt{2\\pi}\\hat{n})|}$$

        As if there is a magnetic pole located just 
        
        $$\\Delta/\\sqrt{2\\pi}$$ 
        
        below the surface
        """
        if not hasattr(self, 'Bpx') or not hasattr(self, 'Bpx') or not hasattr(self, 'Bpz'):

            # Transfer to PyTorch
            Bz = F.pad(torch.from_numpy(self.Bz), (radius, radius, radius, radius)).float()
            Bz = Bz.to(gpu_dev)

            # Distance kernel - a kernel with values filled in the "circle" (by def of norm) as the distance from
            # the center multiplied by dz (for integration)

            Gn = dist_kern.to(gpu_dev)

            # Convolution -- integrate over each pixel
            pot = F.conv2d(Bz[None, None, ...], Gn[None, None, ...])[0][0].cpu().numpy()



            # Save potential
            self.potential = pot

            # Get Potential Fields
            self.Bpz = self.Bz
            grad = gradient(self.potential)
            self.Bpx, self.Bpy = -grad[0], -grad[1]


class ActiveRegionParameters(ActiveRegionFields):

    def __init__(self, Bz, By, Bx, num_features, chosen_funcs = None):
        """A place to define physical parameters of an active region

        Args:
            Bz (2d np.array): z component of magnetic field
            By (2d np.array): y component of magnetic field
            Bx (2d np.array): x component of magnetic field
        """
        super().__init__(Bz, Bx, By)

        if chosen_funcs is None:
            self.chosen_funcs = [   self.Bz_moments,self.Bz_tot,self.Bz_totabs,self.Bh_moments, \
                                    self.gamma_moments,self.GradB_moments,self.GradBz_moments, \
                                    self.GradBh_moments,self.J_moments,self.itot,self.itotabs, \
                                    self.itot_polarity,self.Jh_moments,self.ihtot,self.ihtotabs, \
                                    self.twist_moments,self.hc_moments,self.hctot,self.hctotabs, \
                                    self.shear_moments,self.rho_moments,self.totrho, \
                                    self.entropy]
        else:
            self.chosen_funcs = chosen_funcs

        self.num_features = 0
        for i in self.chosen_funcs:
            if "moments" in i.__name__:
                self.num_features += 4
            else:
                self.num_features += 1

        self.num_features += num_features
        self.labels = []

    def register_func(self, func):
        """Registers a parameter function

        Args:
            func ([type]): A python function that takes in a mask and returns a (label, scalar) or ([label1...labeln], [scalar1...scalarn]) itterables
        """
        self.chosen_funcs.append(func)

    def physical_features(self, mask, labels_prefix = ""):
        """Extracts the physical fetures from a subset of the active region.

        Args:
            mask (np.array): A mask (subset) of the same shape as self (self.shape) 
            where the physical features are computed on (for example, if the mask
            covers a neutral line, then net magnetic flux is calculated as the sum of all
            flux *within that neutral line*)

        Returns:
            np.array: a 1 dimensional array with all of the physical features computed on the subset provided by mask
        """

        data = dict()

        skip = np.count_nonzero(mask) == 0 # Empty

        # 1 dimensional features
        print("=====================")
        for func in self.chosen_funcs:
            name = func.__name__


            if "moment" in name: # Statistical moment
                labels = stat_moment_label(name.split("_")[0])
                if skip:
                    values = [0.0 for _ in labels]
                else:
                    start = time.time()
                    values = func(mask)
                    end = time.time()
                    print(end - start, name)
                for label, value in zip(labels, values):
                    v = float(value)
                    if np.isnan(v):
                        print(f"WARNING: {labels_prefix + label} caused nan")
                    data[labels_prefix + label] = v
                    

            else: # Single value
                label = name
                if skip:
                    value = 0.0
                else:
                    start = time.time()
                    value = float(func(mask))
                    end = time.time()
                    print(end - start, name)

                    v = float(value)
                    if np.isnan(v):
                        print(f"WARNING: {labels_prefix + label} caused nan")
                data[labels_prefix + label] = value

        print("=====================")
        return data

    def entropy(self, mask):
        """Shannon Entropy

        Args:
            mask (array): The mask to compute on

        Returns:
            float: Shannon Entropy of mask
        """
        return shannon_entropy(mask)


    def Bz_moments(self, mask):
        """Statistical moments of the line-of-site magnetic flux (treated as a distribution without regard to x,y coordinates)

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Returns:
            parameter label, $M(B_z)$
        """
        return stat_moment(self.Bz[mask])

    def Bz_tot(self, mask):
        """Sum of the unsigned line of sight flux

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Returns:
            parameter label, $\\sum_{\\Phi\\in B_z}|\\Phi|dA$
        """
        return np.sum(np.abs(self.Bz[mask]))

    def Bz_totabs(self, mask):
        """Unsigned sum of line of sight flux

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Returns:
            parameter label, $|\\sum_{\\Phi\\in B_z}\\Phi dA|$
        """
        return np.abs(np.sum(self.Bz[mask]))

    def Bh_moments(self, mask):
        """Statistical moments of horizontal magnetic field 

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$B_h = |B_x| + |B_y|$$

        Returns:
            parameter label, $M(B_h)$
        """
        self.assert_Bh()
        return stat_moment(self.Bh[mask])

    def gamma_moments(self, mask):
        """Statistical momemnts of angle of magnetic field vector

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$\\gamma = arctan(\\frac{B_z}{|B_x| + |B_y|})$$
        
        Returns:
            parameter label, $M(\\gamma)$
        """
        self.assert_gamma()
        return stat_moment(self.gamma[mask])

    def GradB_moments(self, mask):
        """Statistical moments of the gradient of the magnetic field vector

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            B = norm(B_x, B_y, B_z)

        Returns:
            parameter label, $M(\\nabla B)$
        """
        self.assert_grad_B()
        return stat_moment(self.norm((self.grad_B_x[mask], self.grad_B_y[mask])))

    def GradBz_moments(self, mask):
        """Statistical moments of the gradient of the z component of the magnetic flux

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Returns:
            parameter label, $M(\\nabla B_z)$
        """
        self.assert_grad_Bz()
        return stat_moment(self.norm((self.grad_Bz_x[mask], self.grad_Bz_y[mask])))

    def GradBh_moments(self, mask):
        """Statistical moments of the gradient of the horizontal component of the magnetic flux

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$B_h = norm(B_x, B_y)$$

        Returns:
            parameter label, $M(\\nabla B_h)$
        """
        self.assert_grad_Bh()
        return stat_moment(self.norm((self.grad_Bh_x[mask], self.grad_Bh_y[mask])))
        
    def J_moments(self, mask):
        """Statistical moments of the gradient of the vertical current

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$

        Returns:
            parameter label, $M(J_z)$
        """
        self.assert_J()
        return stat_moment(self.J[mask])

    def itot(self, mask):
        """Sum of unsigned vertical current

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$

        Returns:
            parameter label, $\\sum_{j \\in J_z}|j|dA$
        """
        self.assert_J()
        return np.sum(np.abs(self.J[mask]))

    def itotabs(self, mask):
        """unsigned sum of vertical current

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$

        Returns:
            parameter label, $|\\sum_{j \\in J_z}jdA|$
        """
        self.assert_J()
        return np.abs(np.sum(self.J[mask]))

    def itot_polarity(self, mask):
        """sum of unsigned current regardless of sign of Bz

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$

        Returns:
            parameter label, $|\\sum_{j^+ \\in J_z(B_z > 0)}j^+dA| + |\\sum_{j^- \\in J_z(B_z < 0)}j^-dA|$
        """
        self.assert_J()
        return np.abs(np.sum(self.J[(self.Bz > 0) & mask])) + np.abs(np.sum(self.J[(self.Bz < 0) & mask]))

    def Jh_moments(self, mask):
        """Statistical moments of Vertical heterogeneity current 

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$B = |B_x| + |B_y| + |B_z|$$
            $$J_z^h = \\frac{1}{B}(B_y\\frac{\\partial B_x}{\\partial y} - B_x \\frac{\\partial B_y}{\\partial x})$$

        Returns:
            parameter label, $M(J_z^h)$
        """
        self.assert_Jh()
        return stat_moment(self.Jh[mask])

    def ihtot(self, mask):
        """Sum of unsigned vertical heterogeneity current

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$B = |B_x| + |B_y| + |B_z|$$
            $$J_z^h = \\frac{1}{B}(B_y\\frac{\\partial B_x}{\\partial y} - B_x \\frac{\\partial B_y}{\\partial x})$$

        Returns:
            parameter label, $\\sum_{i\\in J_z^h}|i dA|$
        """
        self.assert_Jh()
        return np.sum(np.abs(self.Jh[mask]))

    def ihtotabs(self, mask):
        """Unsigned Sum of vertical heterogeneity current

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$B = |B_x| + |B_y| + |B_z|$$
            $$J_z^h = \\frac{1}{B}(B_y\\frac{\\partial B_x}{\\partial y} - B_x \\frac{\\partial B_y}{\\partial x})$$

        Returns:
            parameter label, $|\\sum_{i\\in J_z^h}i dA|$
        """
        self.assert_Jh()
        return np.abs(np.sum(self.Jh[mask]))

    def twist_moments(self, mask):
        """Statistical moments of twist

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$
            $$T = \\frac{J_z}{B_z}$$

        Returns:
            parameter label, $M(T)$
        """
        self.assert_J()
        # Assume J(B = 0) = 0 - so twist is "1"
        #JJ = np.nan_to_num(self.J[mask] / self.Bz[mask], nan = 1.0)
        J = self.J[mask]
        Bz = self.Bz[mask]
        
        JJ = np.divide(J, Bz, out = np.ones_like(J), where=Bz!=0)

        return stat_moment(JJ)

    def hc_moments(self, mask):
        """Statistical moments of current helicity

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$h_c = B_z(\\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y})$$

        Returns:
            parameter label, $M(h_c)$
        """
        self.assert_hc()
        return stat_moment(self.hc[mask])

    def hctot(self, mask):
        """Sum of unsigned current helicity

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$h_c = B_z(\\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y})$$

        Returns:
            parameter label, $\\sum_{h \\in h_c}|h|dA$
        """
        self.assert_hc()
        return np.sum(np.abs(self.hc[mask]))

    def hctotabs(self, mask):
        """Unsigned sum of current helicity

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$h_c = B_z(\\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y})$$

        Returns:
            parameter label, $|\\sum_{h \\in h_c}hdA|$
        """
        self.assert_hc()
        return np.abs(np.sum(self.hc[mask]))

    def shear_moments(self, mask):
        """Statistical moments of shear angle

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$\\Psi = arccos(\\frac{\\vec{B_p}\\cdot\\vec{B_o}}{|B_o||B_p|})$$

        Returns:
            parameter label, $M(\\Psi)$
        """
        self.assert_shear() # Pretty big function call right here
        return stat_moment(self.shear[mask])

    def rho_moments(self, mask):
        """Statistical moments of magnetic energy density

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$\\rho_e = |B_p - B_o|$$

        Returns:
            parameter label, $M(\\rho_e)$
        """
        self.assert_rho()
        return stat_moment(self.rho[mask])

    def totrho(self, mask):
        """Total photospheric excess magnetic energy

        Parameterization scalars come from

        **K. D. Leka and G. Barnes. Photospheric magnetic field properties of flaring versus flare-quiet active regions. ii. 
        discriminant analysis. *The Astrophysical Journal* 595(2):1296-1306, 2003.**

        Args:
            mask (np.array): A subset of self to compute upon

        Asserts:
            $$\\rho_e = |B_p - B_o|$$

        Returns:
            parameter label, $$\\sum_{p\\in p_e)pdA$$
        """
        self.assert_rho()
        return np.sum(self.rho[mask])



    def norm(self, data):
        """The one norm of a set of data elements

        Args:
            data (A list, tuple or iterable): A List of elements that can be taken the absolute value of

        Returns:
            The result of summing the application of np.abs() on all elements of data
        """
        return norm(data)
