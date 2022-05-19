# 2 Getting Started
***

## This is a markdown version of ./docs/guides/getting_started.ipynb


```python
# ActiveRegion is the main class - it has access to segmentation methods,
# Physical features and fields
from solarmask.active_region import ActiveRegion

# Usefull if you don't want to manually find datetimes, get_dates
# finds all the dates for a given harpnumber
from solarmask.data import get_dates
```

For any active region, all you need is a **harpnumber** **root** and **date**

Root is the entry point into your data. Root is designed to read from two folders: 

    - root/magnetogram
    - root/continuum
    
In each of these folders, there should be a list of harpnumber directories labeled sharp_\<harpnumber\>:

    - root/magnetogram
        - sharp_1
        - sharp_2
        ...
    - root/magnetogram
        - sharp_1
        - sharp_2
        
In each of these subfolders, there should be a list of active regions. In magnetogram, there is a bz, br and bp component and continuum has a single continuum component using the following naming convention:
    
    - root/magnetogram
        - sharp_1
            - hmi.sharp_cea_720s.<harpnumber>.<yyyy><mm><dd>_<hh><mm><ss>_TAI.Bp.fits
            - hmi.sharp_cea_720s.<harpnumber>.<yyyy><mm><dd>_<hh><mm><ss>_TAI.Br.fits
            - hmi.sharp_cea_720s.<harpnumber>.<yyyy><mm><dd>_<hh><mm><ss>_TAI.Bt.fits
    - root/continuum
        - sharp_1
            - hmi.sharp_cea_720s.<harpnumber>.<yyyy><mm><dd>_<hh><mm><ss>_TAI.continuum.fits

For example:

    - root/magnetogram
        - sharp_7115
            - hmi.sharp_cea_720s.7115.20170903_100000_TAI.Bp.fits
            - hmi.sharp_cea_720s.7115.20170903_100000_TAI.Br.fits
            - hmi.sharp_cea_720s.7115.20170903_100000_TAI.Bt.fits

    - root/continuum
        - sharp_7115
            - hmi.sharp_cea_720s.7115.20170903_100000_TAI.continuum.fits


```python
hnum = 7115 
root = "../example_data/raw"
dates = get_dates(hnum, root, sort = True) # If you don't want to manually put in dates - extract all possible dates

print("First date: ", dates[0])
print("Last date: ", dates[-1])
print("Num dates: ", len(dates))
```

    First date:  2017-09-01 22:00:00
    Last date:  2017-09-06 12:00:00
    Num dates:  2


Then we can create an active region - the entry point into all of the library functions


```python
ar = ActiveRegion(hnum, dates[1], root)
print(dates[1])
```

    2017-09-06 12:00:00


And we can view each of the data elements pulled from the fits file:


```python
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.imshow(ar.cont)
ax1.set_title("Continuum")
ax1.axis(False)

ax2.imshow(ar.Bz)
ax2.set_title("Line of sight magnetic field")
ax2.axis(False)

ax3.imshow(ar.Bh)
ax3.set_title("Horizontal magnetic field")
ax3.axis(False)

fig.set_figheight(10)

plt.show()
```


    
![png](https://raw.githubusercontent.com/SWxTREC/solarmask/master/docs/getting_started/output_7_0.png)



## 2.2 Active Region Basics

An active region is designed using assertions. It automatically updates the data products it needs for various operations. That way, every computationally intensive method is only ever called once and the results are stored in memory. Therefore, there is no need to set up or imply various methods. Although this does mean the execution time of various functions is not always the same.

For example, if I want to create an active region, then get a data product (say baseline), I simply need to call get_baseline. The first call will take a while, but the next call will be instantaneous because AR already has the baseline data set:


```python
import time

ar = ActiveRegion(hnum, dates[1], root)

start = time.time()
ar.segmented_dataset
end = time.time()
print("First time running segmented: ", end - start, " seconds - notice it's large")

start = time.time()
ar.segmented_dataset
end = time.time()
print("Second time running segmented: ", end - start, " seconds - notice it's small because we got it once before")

start = time.time()
ar.baseline_dataset
end = time.time()
print("Getting baseline : ", end - start, " seconds")
```

    First time running segmented:  0.35760951042175293  seconds - notice it's large
    Second time running segmented:  2.288818359375e-05  seconds - notice it's small because we got it once before
    Getting baseline :  0.048726558685302734  seconds


So the only time intensive method is the first one that's called. Everything else is instantaneous. If you want to understand the time ellapsed, you can simply call ar.assert_masks(). This is a high level function that "does everything" to initialize the active region. But you really never need to call this method because every method that needs a mask calls it.

### Visualization

You can show the umbras, penumbras neutral lines backgrounds and graphs side by side with the original image:


```python
ar = ActiveRegion(hnum, dates[1], root)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(ar.cont)
ax1.axis(False)
ax1.set_title("Original Continuum")

ax2.imshow(ar.umbra_mask)
ax2.axis(False)
ax2.set_title("Segmented Umbras")

fig.set_figwidth(20)
fig.set_figwidth(20)

plt.show()
```


    
![png](https://raw.githubusercontent.com/SWxTREC/solarmask/master/docs/getting_started/output_12_0.png)
    



```python
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(ar.cont)
ax1.axis(False)
ax1.set_title("Original Continuum")

ax2.imshow(ar.penumbra_mask)
ax2.axis(False)
ax2.set_title("Segmented Penumbras")

fig.set_figwidth(20)
fig.set_figwidth(20)

plt.savefig("./outputs/penumbras.png")
plt.show()
```


    
![png](https://raw.githubusercontent.com/SWxTREC/solarmask/master/docs/getting_started/output_13_0.png)
    



```python
fig, (ax1, ax2) = plt.subplots(1, 2)

Bz = ar.Bz.copy()
Bz[Bz > 0] = 1
Bz[Bz < 0] = -1
pcm = ax1.imshow(Bz)
ax1.axis(False)
ax1.set_title("Original LOS Magnetogram (Binary Pos and Negative Only)")
fig.colorbar(pcm, ax = ax1)

ax2.imshow(ar.nl_mask)
ax2.axis(False)
ax2.set_title("Segmented Neutral Line")

fig.set_figwidth(20)
fig.set_figwidth(20)

plt.savefig("./outputs/nl.png")
plt.show()
```


    
![png](https://raw.githubusercontent.com/SWxTREC/solarmask/master/docs/getting_started/output_14_0.png)
    



```python
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(ar.cont)
ax1.axis(False)
ax1.set_title("Original Continuum")

ax2.imshow(ar.background_mask)
ax2.axis(False)
ax2.set_title("Segmented Background")

fig.set_figwidth(20)
fig.set_figwidth(20)

plt.savefig("./outputs/background.png")
plt.show()
```


    
![png](https://raw.githubusercontent.com/SWxTREC/solarmask/master/docs/getting_started/output_15_0.png)
    


### Datasets


There are four main datasets that you can get from ActiveRegion:

1. sharps - The 16 (or 32 if INCLUDE_ERRORS in data.py is True) physical features that JSOC extracts and stores in the header file
2. baseline - The 58 physical features I have included run on the entire image
3. segmented - The 58 physical features for each subsection stacked on top of each other (so a 232 dimensional vector)
4. graph - A networkx graph. Each node is connected to its neighbors and each node has an attribute (v) that encodes its physical feature vector (58 dimensions)

Each call to get_<dataset> returns an (array, label) for the array and the label for each value in the array. Note in segmented, if a certain subset doesn't exist, the values in the array will be 0 and the labels will be empty

#### SHARP
Sharp features are those stored in each individual fits file by JSOC

Labels for sharps features are the same as those in JSOC http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm


```python
print(ar.sharps_dataset)
```

    {'USFLUX': 4.533103e+22, 'MEANGAM': 51.792, 'MEANGBT': 110.105, 'MEANGBZ': 126.385, 'MEANGBH': 76.954, 'MEANJZD': -0.31327546, 'TOTUSJZ': 100571700000000.0, 'MEANALP': -0.06190259, 'MEANJZH': -0.03697439, 'TOTUSJH': 7357.481, 'ABSNJZH': 2271.374, 'SAVNCPP': 78371730000000.0, 'MEANPOT': 21630.71, 'TOTPOT': 1.764815e+24, 'MEANSHR': 45.785, 'SHRGT45': 47.17}


#### Baseline
Baseline is the result of calling ar.physical_features on the entire active region 2d array (ie the mask is all true). This is a 58 dimensional vector

Labels for baseline follow the convention:

    - bas_<property>_<moment label (if any)>

For example, the skew of the shear field and the total free energy would be (respectively):

    - bas_shear_skew, bas_totrho


```python
print(ar.baseline_dataset)
```

    {'Bz_tot': 48436571.73000002, 'Bz_totabs': 912665.0699999997, 'itot': 1158731875.8816144, 'itotabs': 513884.65279076196, 'itot_polarity': 69741420.32692963, 'ihtot': 260424630525.3697, 'ihtotabs': 3658852029.034691, 'hctot': 346536044344.56824, 'hctotabs': 152841919638.9422, 'totrho': 21887487.611776374, 'Bz_mean': 2.9610447920992518, 'Bz_std': 353.91205722553934, 'Bz_skew': 0.8624891360407722, 'Bz_kurt': 15.705697123960512, 'Bh_mean': 181.74443637163253, 'Bh_std': 301.11945941453774, 'Bh_skew': 4.275037651713445, 'Bh_kurt': 27.07156202696789, 'gamma_mean': -0.1003125604687583, 'gamma_std': 0.6785771694077445, 'gamma_skew': 0.20137146281062596, 'gamma_kurt': -1.0749836556997958, 'grad_B_mean': 49.819821037258365, 'grad_B_std': 72.48970397795553, 'grad_B_skew': 10.687045428877381, 'grad_B_kurt': 214.83928901171123, 'grad_Bz_mean': 51.636673855447874, 'grad_Bz_std': 81.89618924956946, 'grad_Bz_skew': 9.117129105709143, 'grad_Bz_kurt': 162.31031800893274, 'grad_Bh_mean': 37.43074113475015, 'grad_Bh_std': 59.7594595863308, 'grad_Bh_skew': 14.68324296531682, 'grad_Bh_kurt': 346.624428398112, 'J_mean': 1.667244123724181, 'J_std': 6921.2747549414735, 'J_skew': 5.779373559988659, 'J_kurt': 329.2113441555798, 'Jh_mean': 11870.756427256447, 'Jh_std': 7499866.0437582, 'Jh_skew': 8.102423095067055, 'Jh_kurt': 2704.456210827294, 'twist_mean': -5.383805555755496, 'twist_std': 5431.099285738268, 'twist_skew': 19.27497183475329, 'twist_kurt': 22725.394418643416, 'hc_mean': -495879.35929370264, 'hc_std': 7445779.2243369175, 'hc_skew': -7.804882836278778, 'hc_kurt': 1132.954225325227, 'shear_mean': 1.0360216908634599, 'shear_std': 0.5274806533779264, 'shear_skew': 0.7726592942127313, 'shear_kurt': 0.6151056886795883, 'rho_mean': 1784.7168367601291, 'rho_std': 10948.434868430033, 'rho_skew': 20.25839583283075, 'rho_kurt': 661.463797670551, 'hnum': 7115, 'date': datetime.datetime(2017, 9, 6, 12, 0), 'NOAA_ARS': '12673'}


#### Segmented
Baseline is the result of calling ar.physical_features on each individual segment of the active region and stacking neutral line, umbra, penumbra, background on top of each other

Labels for segmented follow the convention:

    - <region>_<property>_<moment label (if any)>

For example, the skew of the shear field for the neutral line and the total free energy for the background would be (respectively):

    - nl_shear_skew, bckg_totrho


```python
print(ar.segmented_dataset)
```

    {'nl_Bz_tot': 1730154.0, 'nl_Bz_totabs': 74285.20000000001, 'nl_itot': 72911186.77918473, 'nl_itotabs': 5502452.308782724, 'nl_itot_polarity': 8855326.523065047, 'nl_ihtot': 77831129621.29727, 'nl_ihtotabs': 246844235.33597994, 'nl_hctot': 57063866401.77635, 'nl_hctotabs': 12216768812.036888, 'nl_totrho': 6131235.380608724, 'nl_Bz_mean': -18.233971526755035, 'nl_Bz_std': 723.3068414530032, 'nl_Bz_skew': -0.5455602300328546, 'nl_Bz_kurt': 5.615103176236618, 'nl_Bh_mean': 1132.8273410495333, 'nl_Bh_std': 851.3247532920949, 'nl_Bh_skew': 1.189955019812326, 'nl_Bh_kurt': 1.2650226810732885, 'nl_gamma_mean': 0.0018664835082713447, 'nl_gamma_std': 0.5023637771433762, 'nl_gamma_skew': 0.09755952858633657, 'nl_gamma_kurt': 0.6208295788868163, 'nl_grad_B_mean': 250.90027949302583, 'nl_grad_B_std': 349.3953113840282, 'nl_grad_B_skew': 3.0308016114527065, 'nl_grad_B_kurt': 10.88481453737684, 'nl_grad_Bz_mean': 344.88730013572416, 'nl_grad_Bz_std': 383.2043553122473, 'nl_grad_Bz_skew': 2.6043006508825277, 'nl_grad_Bz_kurt': 8.734908003414137, 'nl_grad_Bh_mean': 228.66409790596367, 'nl_grad_Bh_std': 314.4024010027067, 'nl_grad_Bh_skew': 3.119120396916261, 'nl_grad_Bh_kurt': 12.05226641966389, 'nl_J_mean': 1350.626487182799, 'nl_J_std': 36301.3813703652, 'nl_J_skew': 2.194293677572936, 'nl_J_kurt': 21.150018141171014, 'nl_Jh_mean': 60590.1412213991, 'nl_Jh_std': 54303638.36243972, 'nl_Jh_skew': 1.1000197859869298, 'nl_Jh_kurt': 49.72482179690839, 'nl_twist_mean': -61.65084948422465, 'nl_twist_std': 2743.295376656444, 'nl_twist_skew': -34.21663499485756, 'nl_twist_kurt': 1376.7499380434501, 'nl_hc_mean': -2998715.957790105, 'nl_hc_std': 43745259.95469983, 'nl_hc_skew': -0.8086331706840002, 'nl_hc_kurt': 49.02158246393265, 'nl_shear_mean': 1.312623532458661, 'nl_shear_std': 0.6785921010173062, 'nl_shear_skew': 0.39613934510892995, 'nl_shear_kurt': -0.6563335362453024, 'nl_rho_mean': 37823.945074423544, 'nl_rho_std': 65525.4269683639, 'nl_rho_skew': 3.5897674999601406, 'nl_rho_kurt': 18.422030655832334, 'umbra_Bz_tot': 9331012.15, 'umbra_Bz_totabs': 1249786.09, 'umbra_itot': 111803407.93025757, 'umbra_itotabs': 4292295.019404108, 'umbra_itot_polarity': 56418390.142805316, 'umbra_ihtot': 97321262838.26909, 'umbra_ihtotabs': 5082105337.676033, 'umbra_hctot': 153447868010.19266, 'umbra_hctotabs': 93188857576.94708, 'umbra_totrho': 4821409.698779788, 'umbra_Bz_mean': 202.2307588996764, 'umbra_Bz_std': 1611.3267901902395, 'umbra_Bz_skew': -0.12427922699027029, 'umbra_Bz_kurt': -1.5378657517251197, 'umbra_Bh_mean': 989.816124165762, 'umbra_Bh_std': 807.0218224645637, 'umbra_Bh_skew': 1.7154622517116105, 'umbra_Bh_kurt': 2.7119762905670717, 'umbra_gamma_mean': 0.10581819810686431, 'umbra_gamma_std': 1.0755717081793932, 'umbra_gamma_skew': -0.09957603505864095, 'umbra_gamma_kurt': -1.7010370380634439, 'umbra_grad_B_mean': 215.20990983045684, 'umbra_grad_B_std': 310.39938675031004, 'umbra_grad_B_skew': 3.333556391599753, 'umbra_grad_B_kurt': 13.385700916677653, 'umbra_grad_Bz_mean': 250.3393787235854, 'umbra_grad_Bz_std': 336.7864543342683, 'umbra_grad_Bz_skew': 3.2406671106765095, 'umbra_grad_Bz_kurt': 13.3761180411686, 'umbra_grad_Bh_mean': 203.67107608700925, 'umbra_grad_Bh_std': 284.50508557884376, 'umbra_grad_Bh_skew': 3.255946427787143, 'umbra_grad_Bh_kurt': 13.258087574035653, 'umbra_J_mean': 694.5461196446777, 'umbra_J_std': 33443.170473878774, 'umbra_J_skew': 1.9336158882692378, 'umbra_J_kurt': 20.367532971949995, 'umbra_Jh_mean': 822347.1420187756, 'umbra_Jh_std': 45982380.983916186, 'umbra_Jh_skew': 0.42178433137526916, 'umbra_Jh_kurt': 62.255261875977936, 'umbra_twist_mean': -10.995713821906223, 'umbra_twist_std': 404.8042341501762, 'umbra_twist_skew': -21.326298674508042, 'umbra_twist_kurt': 1471.38847273908, 'umbra_hc_mean': -15079103.16779079, 'umbra_hc_std': 45275193.86233077, 'umbra_hc_skew': -0.16372962408572886, 'umbra_hc_kurt': 33.90613282425271, 'umbra_shear_mean': 0.6014057695713156, 'umbra_shear_std': 0.4826412229661533, 'umbra_shear_skew': 1.9452218188249515, 'umbra_shear_kurt': 4.247818017702721, 'umbra_rho_mean': 19607.64438787464, 'umbra_rho_std': 56499.93589951711, 'umbra_rho_skew': 4.777613377124937, 'umbra_rho_kurt': 29.94096538592833, 'penumbra_Bz_tot': 12700424.08, 'penumbra_Bz_totabs': 3230314.2800000003, 'penumbra_itot': 108267025.28137878, 'penumbra_itotabs': 7984027.216685459, 'penumbra_itot_polarity': 33309689.63478607, 'penumbra_ihtot': 75461077924.0739, 'penumbra_ihtotabs': 2593777447.627201, 'penumbra_hctot': 93397266199.10783, 'penumbra_hctotabs': 50537641462.13954, 'penumbra_totrho': 11264777.669405933, 'penumbra_Bz_mean': 189.5057069107122, 'penumbra_Bz_std': 881.3975435745195, 'penumbra_Bz_skew': -0.18535533071855823, 'penumbra_Bz_kurt': -0.7543991001428036, 'penumbra_Bh_mean': 966.1163153303153, 'penumbra_Bh_std': 387.61354509898604, 'penumbra_Bh_skew': 1.2409064845957734, 'penumbra_Bh_kurt': 5.223827820555844, 'penumbra_gamma_mean': 0.14912841290339413, 'penumbra_gamma_std': 0.711764702496566, 'penumbra_gamma_skew': -0.2198079860927122, 'penumbra_gamma_kurt': -0.846557388222045, 'penumbra_grad_B_mean': 101.36553665378653, 'penumbra_grad_B_std': 114.9622603354428, 'penumbra_grad_B_skew': 7.975374352811998, 'penumbra_grad_B_kurt': 111.12473127738106, 'penumbra_grad_Bz_mean': 143.53862847667747, 'penumbra_grad_Bz_std': 134.67439648618864, 'penumbra_grad_Bz_skew': 3.1300004261657244, 'penumbra_grad_Bz_kurt': 17.72996218088503, 'penumbra_grad_Bh_mean': 80.11997970583656, 'penumbra_grad_Bh_std': 110.31153965658159, 'penumbra_grad_Bh_skew': 8.006431221222302, 'penumbra_grad_Bh_kurt': 109.26675504071154, 'penumbra_J_mean': -468.3812751780746, 'penumbra_J_std': 12138.613985964907, 'penumbra_J_skew': 4.448758547177717, 'penumbra_J_kurt': 139.35564332656756, 'penumbra_Jh_mean': 152163.40769841612, 'penumbra_Jh_std': 15444437.15680623, 'penumbra_Jh_skew': 12.207005871823018, 'penumbra_Jh_kurt': 858.970700647569, 'penumbra_twist_mean': -0.39764611473542794, 'penumbra_twist_std': 504.39754922721795, 'penumbra_twist_skew': -43.067305421673076, 'penumbra_twist_kurt': 3965.8839799010143, 'penumbra_hc_mean': -2964780.092815883, 'penumbra_hc_std': 12276917.815575985, 'penumbra_hc_skew': -9.681804603570226, 'penumbra_hc_kurt': 236.29880663697668, 'penumbra_shear_mean': 0.9615597184466406, 'penumbra_shear_std': 0.46696775654873457, 'penumbra_shear_skew': 0.6049189620751559, 'penumbra_shear_kurt': 0.5251511508392306, 'penumbra_rho_mean': 16608.86672323268, 'penumbra_rho_std': 23194.273646323447, 'penumbra_rho_skew': 5.858877508475619, 'penumbra_rho_kurt': 69.13803682789747, 'hnum': 7115, 'date': datetime.datetime(2017, 9, 6, 12, 0), 'NOAA_ARS': '12673'}

