from .utils import *
from .data import *

import torch
import torch.nn.functional as F
import numpy as np


class ActiveRegionDerivedFields:

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
        self.shape = Bz.shape

        # All of the derived fields (private and set to none at first)
        self.__Bz = Bz
        self.__By = By
        self.__Bx = Bx
        self.__Bh = None
        self.__gamma = None
        self.__B = None
        self.__grad_B_x = None
        self.__grad_B_y = None
        self.__grad_B = None
        self.__grad_Bh_x = None
        self.__grad_Bh_y = None
        self.__grad_Bh = None
        self.__grad_Bz_x = None
        self.__grad_Bz_y = None
        self.__grad_Bz = None
        self.__grad_Bx_x = None
        self.__grad_Bx_y = None
        self.__grad_Bx = None
        self.__grad_By_x = None
        self.__grad_By_y = None
        self.__grad_By = None
        self.__J = None
        self.__Jh = None
        self.__hc = None
        self.__twist = None
        self.__shear = None
        self.__rho = None
        self.__Bpx = None
        self.__Bpy = None
        self.__Bpz = None

    def norm(self, data):
        """The one norm of a set of data elements

        Args:
            data (A list, tuple or iterable): A List of elements that can be taken the absolute value of

        Returns:
            The result of summing the application of np.abs() on all elements of data
        """
        return norm(data)


    @property
    def Bz(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__Bz

    @property
    def Bx(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__Bx

    @property
    def By(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__By

    @property
    def Bh(self):
        """Horizontal Magnetic Field Component

        $$B_h = norm(B_x, B_y)$$
        """
        if self.__Bh is None: 
            self.__Bh = self.norm((self.Bx, self.By))
        return self.__Bh

    @property
    def gamma(self):
        """Angle of line of magnetic flux vector from the horizontal

        $$\\gamma = arctan(\\frac{B_z}{|B_x| + |B_y|})$$
        """
        if self.__gamma is None: 
            self.__gamma = np.arctan(self.Bz / self.Bh)
        return self.__gamma

    @property
    def B(self):
        """Magnitude of magnetic flux vector

        $$B = norm(B_x, B_y, B_z)$$
        """
        if self.__B is None:
            self.__B = self.norm((self.Bx, self.By, self.Bz))
        return self.__B

    @property
    def grad_B_x(self):
        """Gradient of magnetic field magnitude

        $$B = norm(B_x, B_y, B_z)$$
        """
        if self.__grad_B_x is None or self.__grad_B_y is None:
            self.__grad_B_x, self.__grad_B_y = gradient(self.B)
        return self.__grad_B_x

    @property
    def grad_B_y(self):
        """Gradient of magnetic field magnitude

        $$B = norm(B_x, B_y, B_z)$$
        """
        if self.__grad_B_x is None or self.__grad_B_y is None:
            self.__grad_B_x, self.__grad_B_y = gradient(self.B)
        return self.__grad_B_y

    @property
    def grad_B(self):
        if self.__grad_B is None:
            self.__grad_B = self.norm((self.grad_B_x, self.grad_B_y))
        return self.__grad_B

    @property
    def grad_Bh_x(self):
        """Gradient of horizontal magnetic field

        $$B_h = norm(B_x, B_y)$$
        """
        if self.__grad_Bh_x is None or self.__grad_Bh_y is None:
            self.__grad_Bh_x, self.__grad_Bh_y = gradient(self.Bh)
        return self.__grad_Bh_x

    @property
    def grad_Bh_y(self):
        """Gradient of horizontal magnetic field

        $$B_h = norm(B_x, B_y)$$
        """
        if self.__grad_Bh_x is None or self.__grad_Bh_y is None:
            self.__grad_Bh_x, self.__grad_Bh_y = gradient(self.Bh)
        return self.__grad_Bh_y

    @property
    def grad_Bh(self):
        if self.__grad_Bh is None:
            self.__grad_Bh = self.norm((self.grad_Bh_x, self.grad_Bh_y))
        return self.__grad_Bh

    @property
    def grad_Bz_x(self):
        """Gradient of line of site magnetic field
        """
        if self.__grad_Bz_x is None or self.__grad_Bz_y is None:
            self.__grad_Bz_x, self.__grad_Bz_y = gradient(self.Bz)
        return self.__grad_Bz_x

    @property
    def grad_Bz_y(self):
        """Gradient of line of site magnetic field
        """
        if self.__grad_Bz_x is None or self.__grad_Bz_y is None:
            self.__grad_Bz_x, self.__grad_Bz_y = gradient(self.Bz)
        return self.__grad_Bz_y

    @property
    def grad_Bz(self):
        if self.__grad_Bz is None:
            self.__grad_Bz = self.norm((self.grad_Bz_x, self.grad_Bz_y))
        return self.__grad_Bz

    @property
    def grad_Bx_x(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_Bx_x is None or self.__grad_Bx_y is None:
            self.__grad_Bx_x, self.__grad_Bx_y = gradient(self.Bx)
        return self.__grad_Bx_x

    @property
    def grad_Bx_y(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_Bx_x is None or self.__grad_Bx_y is None:
            self.__grad_Bx_x, self.__grad_Bx_y = gradient(self.Bx)
        return self.__grad_Bx_y

    @property
    def grad_Bx(self):
        if self.__grad_Bx is None:
            self.__grad_Bx = self.norm((self.grad_Bx_x, self.grad_Bx_y))
        return self.__grad_Bx

    @property
    def grad_By_x(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_By_x is None or self.__grad_By_y is None:
            self.__grad_By_x, self.__grad_By_y = gradient(self.Bx)
        return self.__grad_By_x

    @property
    def grad_By_y(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_By_x is None or self.__grad_By_y is None:
            self.__grad_By_x, self.__grad_By_y = gradient(self.Bx)
        return self.__grad_By_y

    @property
    def grad_By(self):
        if self.__grad_By is None:
            self.__grad_By = self.norm((self.grad_By_x, self.grad_By_y))
        return self.__grad_By

    @property
    def J(self):
        """Vertical current density

        $$J_z = \\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y}$$
        """
        if self.__J is None:
            self.__J = (self.grad_By_x - self.grad_Bx_y) / mu0
            self.__J[self.Bz == 0] = 0
        return self.__J

    @property
    def Jh(self):
        """Vertical heterogeneity current density

        $$J_z^h = \\frac{1}{B}(B_y\\frac{\\partial B_x}{\\partial y} - B_x \\frac{\\partial B_y}{\\partial x})$$
        """
        if self.__Jh is None:
            self.__Jh = (self.By * self.grad_Bx_y - self.Bx * self.grad_By_x) / mu0
        return self.__Jh

    @property
    def hc(self):
        """Current helicity

        $$h_c = B_z(\\frac{\\partial B_y}{\\partial x} - \\frac{\\partial B_x}{\\partial y})$$
        """
        if self.__hc is None:
            self.__hc = self.Bz * self.J
        return self.__hc

    @property
    def twist(self):
        """Twist
        """
        if self.__twist is None:
            self.__twist = np.divide(self.J, self.Bz, out = np.ones_like(self.J), where=self.Bz != 0)
        return self.__twist

    @property
    def shear(self):
        """Shear angle

        $$\\Psi = arccos(\\frac{\\vec{B_p}\\cdot\\vec{B_o}}{|B_o||B_p|})$$
        """
        if self.__shear is None:
            dot = self.Bx * self.Bpx + self.By * self.Bpy + self.Bz * self.Bpz
            magp = self.norm((self.Bpx, self.Bpy, self.Bpz))
            self.__shear = np.arccos(dot / (self.B * magp))
        return self.__shear

    @property
    def rho(self):
        """Excess magnetic energy density

        $$\\rho_e = norm(B_p - B_o)$$
        """
        if self.__rho is None:
            self.__rho = (self.B - self.norm((self.Bpx, self.Bpy, self.Bpz)))**2 / (8*np.pi)
        return self.__rho

    @property
    def Bpx(self):
        if self.__Bpx is None:
            self.greenpot()
        return self.__Bpx

    @property
    def Bpy(self):
        if self.__Bpy is None:
            self.greenpot()
        return self.__Bpy

    @property
    def Bpz(self):
        if self.__Bpz is None:
            self.__Bpz = self.__Bz
        return self.__Bpz

    def greenpot(self):
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
        # Transfer to PyTorch
        Bz = F.pad(torch.from_numpy(self.Bz), (radius, radius, radius, radius)).float()
        Bz = Bz.to(gpu_dev)

        # Distance kernel - a kernel with values filled in the "circle" (by def of norm) as the distance from
        # the center multiplied by dz (for integration)

        Gn = dist_kern.to(gpu_dev)

        # Convolution -- integrate over each pixel
        pot = F.conv2d(Bz[None, None, ...], Gn[None, None, ...])[0][0].cpu().numpy()

        # Save potential
        self.__potential = pot

        # Get Potential Fields
        grad = gradient(self.__potential)
        self.__Bpx, self.__Bpy = -grad[0], -grad[1]



