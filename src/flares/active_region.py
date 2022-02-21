from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from flares.utils import *
from flares.data import *
from flares.fields import ActiveRegionParameters

import torch
import torch.nn.functional as F
from skimage.morphology import square, binary_dilation
from skimage.measure import label
import numpy as np
import networkx as nx
import warnings

from skimage.filters import threshold_local
from skimage.morphology import square
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm



class ActiveRegion(ActiveRegionParameters):
    def __init__(self, hnum: int, date: datetime, root: str, num_features = 0, cmap = "viridis"):
        """An Active Region is an entry point into 
        parameterization, segmentation, and graph methods. 

        An Active Region is uniquely determined by it's harpnumber (hnum) and date (date)
        If the specified active region given (hnum, date) contains a nan, self.valid is False
        (otherwise self.valid is True) and subsequent calls to ActiveRegion methods may fail 
        given the existence of nans. Check status of the active region by self.valid.
        The existence of nan's implies the image is moving off the solar limb. This is an issue that
        could be turned into a ticket and fixed, but for the time being I am ignoring these images

        Magnetograms and continuums must be organized by the following standard (with root as the root)
        (This standard may be loosened in the future and subsequent documentation will change)

        - root
            - magnetogram
                - sharp_<hnum>
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.Bp.fits
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.Br.fits
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.Bt.fits
            - continuum
                - sharp_<hnum>
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.continuum.fits
        
        So for example, (this will work on Swami), a new active region call:
        ```python
        ActiveRegion(7115, datetime(2017, 9, 3, 10), "/srv/data/thli2739") 
        ```

        Would **require** the following files or symlinks to exist:

        - /srv/data/thli2739
            - /magnetogram
                - /sharp_7115
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.Bp.fits
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.Br.fits
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.Bt.fits
            - /continuum
                - /sharp_7115
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.continuum.fits

        Args:
            hnum (int): The specified harpnumber - file must exist in root/magnetogram/sharp_{hnum} and root/continuum/sharp_{hnum}
            date (datetime): The specified active region date and time - this date must exist in the specified harpnumber data folder 
            root (string): The path to the data. Root must be a directory that holds both root/magnetogram and root/continuum. Inside both
            of these subfolders, there must be a series of folders labeled sharp_{hnum} that contain the sequence of fits files for extraction
        """

        self.cmap = cmap
    
        # Generate xyz components of magnetic field and continuum
        data = get_data(hnum, date, root)

        self.Bz, self.Bx, self.By, self.cont = data["Bz"], data["Bx"], data["By"], data["cont"]
        
        self.Bz[np.abs(self.Bz) < 0.001] = 0.0
        self.shape = self.Bz.shape
        self.valid = True # Valid is false


        if np.count_nonzero(np.isnan(self.Bz)) / self.Bz.size > 0.0:
            x, y = np.argwhere(np.isnan(self.Bz)).T
            minx, maxx = np.min(x), np.max(x)
            miny, maxy = np.min(y), np.max(y)

            self.valid = False
            warnings.warn(f"Hnum {hnum} date {date} has a nan, skipping")
            return

        # Now Bx By Bz are defined so generate the parameter class
        super().__init__(self.Bz, self.By, self.Bx, num_features)

        # The four "segments" as raw arrays (no nodes for graph)
        self.__background = None
        self.__umbra = None
        self.__pumbra = None
        self.__nl = None

        # The Three data sets
        self.__sharps = data["sharps"]
        self.__baseline = dict()
        self.__segmented = dict()
        self.__G = nx.Graph()
        self.__G_labels = None

        # GRAPH DATA
        # Graph data is split between masks and feature vectors. Each "mask" is a node that gets its physical features computed on
        self.__node_masks = np.zeros((0, self.shape[0], self.shape[1]), dtype = bool)


        # Flags - to check and examine by eye?
        self.__flags = {"bordered_umbras" : False}
        

    def show_graph(self, axs_cont, axs_seg):
        """Plots the continuum next to segmented graph with nodes
        Located on the graph with colors
        
        Note that umbras and penumbras are generally going to be very close to one another
        This is because the umbra is near the center of the penumbra so their centers are going
        to be close - there are a few good examples of nodes that work well - see documentation/guides

        Args:
            axs_cont (axis): The continuum Axis 
            axs_seg (axis): The segmented axis
        """
        self.assert_masks()

        color_keys = {"penumbra" : "red", "umbra" : "green", "neutral line" : "blue"}
        values = [color_keys[x[1]["type"]] for x in self.__G.nodes.data()]
        pos = nx.get_node_attributes(self.__G, "pos")

        # Plot with the original continuum
        axs_cont.imshow(self.cont, cmap = self.cmap)
        nx.draw(self.__G, pos, axs_cont, node_size = 100, cmap = self.cmap, node_color = values, with_labels = False, font_color = "white")

        # Plot the one next to it
        mask = np.zeros(self.__umbra.shape)
        mask[self.__umbra] = 1
        mask[self.__pumbra] = 2
        mask[self.__nl] = 3
        mask[self.__background] = np.nan
        axs_seg.imshow(mask, cmap = self.cmap)
        nx.draw(self.__G, pos, axs_seg, node_size = 100, cmap = self.cmap, node_color = values, with_labels = False, font_color = "white")


        cmap = cm.get_cmap(self.cmap)
        l1 = mpatches.Patch(color=color_keys["penumbra"], label = "Penumbra")
        l2 = mpatches.Patch(color=color_keys["umbra"], label = "Umbra")
        l3 = mpatches.Patch(color=color_keys["neutral line"], label = "Neutral Line")
        axs_seg.legend(handles=[l1, l2, l3])


    def draw_graph(self, axs):
        """Draws just the graph without a background image of the continuum

        Args:
            axs (pyplot.axis): The axis to write the graph to
        """
        self.assert_masks()

        color_keys = {"penumbra" : "red", "umbra" : "green", "neutral line" : "blue"}
        values = [color_keys.get(x[1]["type"], 0.25) for x in self.__G.nodes.data()]
        pos = nx.get_node_attributes(self.__G, "pos")
        pos = {i : (pos[i][0], -pos[i][1]) for i in pos} # Flip them because of imshow

        nx.draw(self.__G, pos, axs, node_size = 100, cmap = self.cmap, node_color = values, with_labels = False, font_color = "white")

        cmap = cm.get_cmap(self.cmap)
        l1 = mpatches.Patch(color=color_keys["penumbra"], label = "Penumbra")
        l2 = mpatches.Patch(color=color_keys["umbra"], label = "Umbra")
        l3 = mpatches.Patch(color=color_keys["neutral line"], label = "Neutral Line")
        axs.legend(handles=[l1, l2, l3])


    def show_umbra(self, axs_orig, axs_seg):
        """Shows the umbra segmented next to the continuum

        Args:
            axs_orig (matplotlib axis): The axis for the unsegmented image
            axs_seg (matplotlib axis): The axis for the segmented image
        """
        self.assert_masks()
        axs_orig.imshow(self.cont, cmap = "gray")
        axs_seg.imshow(~self.__umbra, cmap = "gray")
        axs_orig.axis(False)
        axs_seg.axis(False)
    
    def show_neutral_line(self, axs_orig, axs_seg):
        """Shows the neutral line segmented next to the line of sight magnetic field

        Args:
            axs_orig (matplotlib axis): The axis for the unsegmented image
            axs_seg (matplotlib axis): The axis for the segmented image
        """
        self.assert_masks()
        axs_orig.imshow(self.Bz, cmap = "gray")
        axs_seg.imshow(~self.__nl, cmap = "gray")
        axs_orig.axis(False)
        axs_seg.axis(False)

    def show_penumbra(self, axs_orig, axs_seg):
        """Shows the penumbra segmented next to the continuum

        Args:
            axs_orig (matplotlib axis): The axis for the unsegmented image
            axs_seg (matplotlib axis): The axis for the segmented image
        """
        self.assert_masks()
        axs_orig.imshow(self.cont, cmap = "gray")
        axs_seg.imshow(~self.__pumbra, cmap = "gray")
        axs_orig.axis(False)
        axs_seg.axis(False)

    def show_background(self, axs_orig, axs_seg):
        """Shows the background segmented next to the continuum

        Args:
            axs_orig (matplotlib axis): The axis for the unsegmented image
            axs_seg (matplotlib axis): The axis for the segmented image
        """
        self.assert_masks()
        axs_orig.imshow(self.cont, cmap = "gray")
        axs_seg.imshow(~self.__background, cmap = "gray")
        axs_orig.axis(False)
        axs_seg.axis(False)

    def get_graph(self):
        """Gets the sharps data and labels

        Sharps data is the data JSOC calculates - good baseline

        Returns:
            data, labels
        """
        self.assert_masks()
        return self.__G, self.__G_labels

    def get_sharps(self):
        """Gets the sharps data and labels

        Sharps data is the data JSOC calculates - good baseline

        Returns:
            data, labels
        """
        self.assert_masks()
        return self.__sharps

    def get_baseline(self):
        """Get function for baseline

        Returns:
            numpy array: baseline data set
        """
        self.assert_masks()
        if len(self.__baseline) == 0:
            self.__baseline = self.physical_features(np.ones(self.shape, dtype = bool), "bas_")
        return self.__baseline

    def get_segmented(self):
        """Get function for segmented

        Returns:
            numpy array: segmented data set 
        """
        self.assert_masks()
        return self.__segmented

    def assert_masks(self):
        """One function that generates all three masks. If all three masks are already
        generated, this function does nothing
        """
        self.assert_neutral_lines()
        self.assert_umbra_pumbra()
        self.assert_background()
    
    def assert_background(self):
        """Generates a background mask. If background is already generated, does nothing
        Background is simply $$\\neg (Umbra \\cup Penumbra \\cup Neutral Line)$$
        """
        if self.__background is None:
            self.__background = np.zeros(self.shape, dtype=bool)
            self.assert_neutral_lines()
            self.assert_umbra_pumbra()

            background = ~(self.__nl | self.__umbra | self.__pumbra)

            # Update dataset
            self.__segmented.update(self.physical_features(background, "bckg_"))
            self.__background = background

    def assert_neutral_lines(self, radius = 5, thresh = 150):
        """Generates neutral lines using morphological operations and a flux threshold
        finds neutral lines using the method described by Schrijver in
        
        **C. J. Schrijver. A characteristic magnetic field pattern associated with all major solar flares and its use in flare forecasting. *The Astrophysical Journal, 655(2), 2007.***


        Args:
            radius (int, optional): The found neutral line is one pixel thick, so dilating it slightly adds the surrounding neighborhood. This radius
            is passed to sklearn.morphology.square(radius) in the call to sklearn.morphology.binary_dilation. Defaults to 3
            thresh (int, optional): Flux threshold. Defaults to 150 (as used by Schjriver et. al).
        """
        if self.__nl is None:
            # Find neutral Lines


            ######### SEGMENTED DATA SET ############
            nl_mask = binary_dilation(self.Bz < -thresh, square(radius)) & binary_dilation(self.Bz > thresh, square(radius))
            self.__nl = nl_mask.copy()

            # Compute the segmented data set
            self.__segmented.update(self.physical_features(nl_mask, "nl_"))

            ######### GRAPH DATA SET ################
            labeled, labels, sizes = self.__group_pixels(nl_mask)
            labels, sizes = self.__remove_small_groups(labeled, labels, sizes, 500)
            labels, sizes = self.__remove_percentage_max(labeled, labels, sizes)
            labels, sizes = self.__largest_n_clusters(labels, sizes)

            # Add all the graph nodes
            cur_node = len(self.__node_masks)

            for i in labels:
                mask = labeled == i
                cur_node = self.__ar_add_node(cur_node, mask, "neutral line")

            
            

    

    def assert_umbra_pumbra(self):
        """An original algorithm for detecting umbras and penumbras from a continuum image

        High Level Algorithm:

        1. bound continuum between 0 and 255

        2. use an adaptive filter on the bounded continuum

        3. group and label touching pixels 

        4. remove groups of pixels that are less than 500 from those remaining (if any) 

        5. remove groups of pixels that border the image (usually noise) from those remaining (if any) 

        6. remove all groups that are smaller than 10% of the size of the maximum group size from those remaining (if any)  

        7. remove take the largest 6 clusters from those remaining (if any) 

        8. The remaining groups are **penumbra outlines**, repeat the above process isolated only to the penumbra outlines
        if the difference between maximum and minimum flux in the mask is greater than 21000 and the resulting clusters are umbras

        9. Keep the remaining 6 largest umbras
        """
        if self.__pumbra is None or self.__umbra is None:
            #We first segment large groups (that may be penumbras or umbras)
            cont_bounded = (255 * (self.cont - np.min(self.cont)) / np.ptp(self.cont)).astype(np.uint8)
 
            block_size = np.min(self.shape)
            if block_size % 2 == 0:
                block_size -= 1
                
            offset = 10
            binary_adaptive = cont_bounded < (threshold_local(cont_bounded, block_size, offset = offset) - offset)

            ###### Filter Segmented ###########
            labeled_0, labels, sizes = self.__group_pixels(binary_adaptive)
            labels, sizes = self.__remove_bordering_pixels(labeled_0, labels, sizes)

            ###### UMBRA SEGMENTED MASK #############
            self.__umbra = (np.isin(labeled_0, labels))
            
            ###### Filter for graph #######
            labels, sizes = self.__remove_small_groups(labeled_0, labels, sizes)
            labels, sizes = self.__remove_percentage_max(labeled_0, labels, sizes)
            labels, sizes = self.__largest_n_clusters(labels, sizes)

            self.__pumbra = np.zeros(self.shape, dtype = bool)

            # For each large group - determine if this is a penumbra / umbra combo or just umbra
            cur_node = len(self.__node_masks)
            for i in labels:
                mask = labeled_0 == i
                mx = np.max(self.cont[mask])
                mn = np.min(self.cont[mask])
                t = (mx - mn) / 2

                # PENUMBRA AND UMBRA
                if mx - mn > 21000:
                    # Both umbra and penumbra
                    um = mask & (self.cont <= t)
                    pu = mask & (self.cont > t)



                    # Further segment the umbra node again
                    labeled, labels, sizes = self.__group_pixels(pu)
                    labels, sizes = self.__remove_small_groups(labeled_0, labels, sizes, 3)
                    labels, sizes = self.__remove_percentage_max(labeled_0, labels, sizes)
                    labels, sizes = self.__largest_n_clusters(labels, sizes)

                    for i in labels:
                        mask = labeled == i
                        cur_node = self.__ar_add_node(cur_node, mask, "penumbra")
                        self.__pumbra |= mask
                        self.__umbra &= ~mask

                    # Further segment the umbra node again
                    labeled, labels, sizes = self.__group_pixels(um)
                    labels, sizes = self.__remove_small_groups(labeled_0, labels, sizes, 3)
                    labels, sizes = self.__remove_percentage_max(labeled_0, labels, sizes)
                    labels, sizes = self.__largest_n_clusters(labels, sizes)

                    for i in labels:
                        mask = labeled == i
                        cur_node = self.__ar_add_node(cur_node, mask, "umbra")

                # ONLY UMBRA
                else:
                    um = mask & (self.cont <= t)
                    cur_node = self.__ar_add_node(cur_node, mask, "umbra")
            
            
            self.__segmented.update(self.physical_features(self.__umbra, "um_"))
            self.__segmented.update(self.physical_features(self.__pumbra, "pu_"))


    def __group_pixels(self, mask):
        """Groups pixels in a binary mask based on if they are touching

        Args:
            mask (np.array): A mask with binary pixels

        Returns:
            labeled: a mask that is labeled from 0 to number of groups (0 is the background) the size of the number in the label doesn't mean anything
            labels: a list of labels in labeled
            sizes: a list of sizes corresponding to each element in labels
        """
        labeled = label(mask, connectivity = 2)
        labels = np.unique(labeled)[1:]
        sizes = np.array([np.count_nonzero(labeled==i) for i in labels])
        return labeled, labels, sizes

    def __remove_small_groups(self, labeled: np.array, labels: np.array, sizes: np.array, p = 500):
        """Removes groups of pixels smaller than p

        Args:
            labeled (np.array): The labeled array returned from group_pixels
            labels (np.array): The distinct labels found in labeled
            sizes (np.array): The size of each label in labels
            p (int, optional): Smallest group size. Defaults to 500.

        Returns:
            labels: Labels filtered 
            sizes: sizes filtered 
        """
        if len(sizes) == 0:
            return labels, sizes
        filt = np.argwhere((sizes < p))
        return np.delete(labels, filt), np.delete(sizes, filt)

    def __remove_bordering_pixels(self, labeled: np.array, labels: np.array, sizes: np.array):
        """Removes pixels that border the edge

        Args:
            labeled (np.array): The labeled array returned from group_pixels
            labels (np.array): The distinct labels found in labeled
            sizes (np.array): The size of each label in labels
            p (int, optional): Smallest group size. Defaults to 500.

        Returns:
            labels: Labels filtered 
            sizes: sizes filtered 
        """
        if len(sizes) == 0:
            return labels, sizes
        bordered = []
        for i in range(len(labels)):
            rows, cols = np.where(labeled == labels[i])
            if min(rows) == 0 or min(cols) == 0:
                bordered.append(i)
                self.__flags["bordered_umbras"] = True
            if max(cols) == self.shape[1] - 1 or max(rows) == self.shape[0] - 1:
                bordered.append(i)
                self.__flags["bordered_umbras"] = True
        return np.delete(labels, bordered), np.delete(sizes, bordered)

    def __remove_percentage_max(self, labeled, labels, sizes, p = 0.01):
        """Removes pixels that border the edge

        Args:
            labeled (np.array): The labeled array returned from group_pixels
            labels (np.array): The distinct labels found in labeled
            sizes (np.array): The size of each label in labels
            p (int, optional): Smallest group size. Defaults to 500.

        Returns:
            labels: Labels filtered 
            sizes: sizes filtered 
        """
        if len(sizes) == 0:
            return labels, sizes
        filt = np.argwhere(sizes < p * np.max(sizes))
        return np.delete(labels, filt), np.delete(sizes, filt)

    def __largest_n_clusters(self, labels, sizes, n = 6):
        """Removes pixels that border the edge

        Args:
            labeled (np.array): The labeled array returned from group_pixels
            labels (np.array): The distinct labels found in labeled
            sizes (np.array): The size of each label in labels
            p (int, optional): Smallest group size. Defaults to 500.

        Returns:
            labels: Labels filtered 
            sizes: sizes filtered 
        """
        if len(sizes) == 0:
            return labels, sizes
        n = min(n, len(labels))
        a = np.partition(sizes, -n)[-n]
        return labels[sizes >= a], sizes[sizes >= a]

    def __ar_add_node(self, cur_node, mask, type):
        """Adds a node to self graph and connects to all the previous nodes

        Args:
            data ([type]): [description]
            cur_node ([type]): [description]
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """

        data = self.physical_features(mask, "graph_")
        if self.__G_labels is None:
            self.__G_labels = list(data.keys())
        data = data.values()

        # Get the center of the node for plotting
        x, y = np.where(mask)
        x, y = int(np.mean(x)), int(np.mean(y))
        self.__G.add_node(cur_node, v = data, pos = (y,x), type = type)
        #self.__G.add_node(cur_node, v = data, pos = (0,100), type = type)

        mask_dil = binary_dilation(mask, square(3))

        for m in range(len(self.__node_masks)):
            if np.count_nonzero(mask_dil & self.__node_masks[m]) > 0:
                self.__G.add_edge(cur_node, m)

        self.__node_masks = np.concatenate((self.__node_masks, mask_dil[None,...]), axis = 0)
        return cur_node + 1

    def get_node_masks(self):
        """Getter for node masks

        Returns:
            np.array: A list of masks lining up to each of the node indexes / node names
        """
        return self.__node_masks

    


