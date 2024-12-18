# atomistic-water-spectra
 
Contains code to generate vibrational Hamiltonian, transition dipole, and transition polarizability trajectories in binary file formats to be passed to [NISE](https://github.com/GHlacour/NISE_2017) for final vibrational spectra calculations  

#### Based on work by James L. Skinner and coworkers
1. Gruenbaum, S. M.; Tainter, C. J.; Shi, L.; Skinner, J. L. Robustness of Frequency, Transition Dipole, and Coupling Maps for Water Vibrational Spectroscopy. *J. Chem. Theory Comput.* **2013**, *9* (7), 3109–3117. [https://doi.org/10.1021/ct400292q.](https://doi.org/10.1021/ct400292q)  
2. Auer, B. M.; Skinner, J. L. IR and Raman Spectra of Liquid Water: Theory and Interpretation. *J. Chem. Phys.* **2008**, *128* (22), 224511. [https://doi.org/10.1063/1.2925258.](https://doi.org/10.1063/1.2925258)
3. Kananenka, A. A.; Skinner, J. L. Fermi Resonance in OH-Stretch Vibrational Spectroscopy of Liquid Water and the Water Hexamer. *J. Chem. Phys.* **2018**, *148* (24), 244107. [https://doi.org/10.1063/1.5037113.](https://doi.org/10.1063/1.5037113)
4. Pieniazek, P. A.; Tainter, C. J.; Skinner, J. L. Surface of Liquid Water: Three-Body Interactions and Vibrational Sum-Frequency Spectroscopy. *J. Am. Chem. Soc.* **2011**, *133* (27), 10360–10363. [https://doi.org/10.1021/ja2026695.](https://doi.org/10.1021/ja2026695)
5. Ni, Y.; Skinner, J. L. Communication: Vibrational Sum-Frequency Spectrum of the Air-Water Interface, Revisited. *J. Chem. Phys.* **2016**, *145* (3), 031103. [https://doi.org/10.1063/1.4958967.](https://doi.org/10.1063/1.4958967)
6. Kananenka, A. A.; Yao, K.; Corcelli, S. A.; Skinner, J. L. Machine Learning for Vibrational Spectroscopic Maps. *J. Chem. Theory Comput.* **2019**, *15* (12), 6850–6858. [https://doi.org/10.1021/acs.jctc.9b00698.](https://doi.org/10.1021/acs.jctc.9b00698)



## Usage

Two .yml files are available. ```water-spectra.yml``` contains the environment needed to run the code using the standard electric field spectroscopic maps. ```water-spectra-delta-ML.yml``` contains the environment needed to run the code using the ∆-ML corrected spectroscopic maps. The other files associated with those maps are located in the ```delta-ML-maps``` directory.

Run the following command to see details of different options.  
``` python gen_ham.py -h ```  

For an overview of compatible file formats, see the [MDAnalysis Format overview](https://userguide.mdanalysis.org/stable/formats/index.html) page. 
