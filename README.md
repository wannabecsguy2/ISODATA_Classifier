# ISODATA_Classifier
Python Based Implementation of the ISODATA Classifier with GUI enabled

GNR-602 Advanced Image Processing
Group - 68
-----------------------------------------------------------------------------------------------------------------

We have taken Indian Pines dataset. This scene was gathered by AVIRIS sensor over the Indian Pines test site in
North-western Indiana and consists of 145´145 pixels and 224 spectral reflectance bands in the wavelength range 
0.4–2.5 10-6 meters. The Indian Pines scene contains two-thirds agriculture, and one-third forest or other natural 
perennial vegetation. There are two major dual lane highways, a rail line, as well as some low density housing,
other built structures, and smaller roads.

------------------------------------------------------------------------------------------------------------------

The dataset is downloaded on system which can be imported from local directory to run the code.

Code files - GUI.py
             Classifier.py
             Executable.py
------------------------------------------------------------------------------------------------------------------

Steps to run the program -

1. Execute and compile "Executable.py" which will call "GUI.py" which will further call "Classifier.py" to run the
program.

-------------------------------------------------------------------------------------------------------------------

Parameters chosen - 

1. Desired number of clusters - 16
2. Maximum number of iterations - 100
3. Maximum clusters pair mergers - 1
4. Number of Clusters starting pair - 16
5. Threshold's cluster size - 10
6. Threshold for Intraclass standard deciation - 100
7. Threshold for Pairwise Distances - 2000
8. Threshold for Consecutive Iteration Change in Cluster - 0.05
