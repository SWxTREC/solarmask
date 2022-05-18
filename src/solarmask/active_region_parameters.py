from .active_region_derived_fields import ActiveRegionDerivedFields
from .utils import *
import numpy as np
import warnings
import time


class ActiveRegionParameters(ActiveRegionDerivedFields):

    def __init__(self, Bz, By, Bx):
        """A place to define physical parameters of an active region

        Args:
            Bz (2d np.array): z component of magnetic field
            By (2d np.array): y component of magnetic field
            Bx (2d np.array): x component of magnetic field
        """
        super().__init__(Bz, Bx, By)

        # All the fields we want statistical moments from
        self.stat_moments = [   "Bz", "Bh", "gamma", "grad_B", "grad_Bz", "grad_Bh", \
                                "J", "Jh", "twist", "hc", "shear", "rho"]

        # All the scalar features derived from derived fields
        self.scalar_features = [self.Bz_tot, self.Bz_totabs, self.itot, self.itotabs, self.itot_polarity, \
                                self.ihtot, self.ihtotabs, self.hctot, self.hctotabs, self.totrho]

        self.num_features = len(self.stat_moments) * 4 + len(self.scalar_features)


    def physical_features(self, mask, labels_prefix = ""):
        """Extracts the physical fetures from a subset of the active region.

        Args:
            mask (np.array): A mask (subset) of the same shape as self (self.shape) 
            where the physical features are computed on (for example, if the maskB
            covers a neutral line, then net magnetic flux is calculated as the sum of all
            flux *within that neutral line*)

        Returns:
            np.array: a 1 dimensional array with all of the physical features computed on the subset provided by mask
        """

        data = dict()
        skip = np.count_nonzero(mask) == 0 # Empty

        # Get all the scalar features
        for func in self.scalar_features:
            label = func.__name__
            if skip:
                v = 0.0
            else:
                start = time.time()
                value = func(mask)
                v = float(value)
                if np.isnan(v):
                    warn = f"{labels_prefix + label} caused nan"
                    warnings.warn(warn)
            data[labels_prefix + label] = v


        # Get all the statistical moments
        for name in self.stat_moments:
            start = time.time()
            field = getattr(self, name)            
            labels = stat_moment_label(name)
            if skip:
                values = [0.0 for _ in labels]
            else:
                values = stat_moment(field[mask])

            for label, value in zip(labels, values):
                v = float(value)
                if np.isnan(v):
                    warn = f"{labels_prefix + name} caused nan"
                    warnings.warn(warn)
                data[labels_prefix + label] = v

        return data

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
        return np.abs(np.sum(self.J[(self.Bz > 0) & mask])) + np.abs(np.sum(self.J[(self.Bz < 0) & mask]))

    
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
        return np.abs(np.sum(self.Jh[mask]))

    
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
        return np.abs(np.sum(self.hc[mask]))

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
        return np.sum(self.rho[mask]) / (8 * np.pi)
