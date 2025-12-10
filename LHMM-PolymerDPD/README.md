# LHMM - Polymer DPD
Fully Lagrangian Heterogeneous Multiscale Method implemented in LAMMPS (SPH-DPD) 

<p>
<img src="docs/figs/workflow.pdf" width="900">
</p>

## Description

A fully Lagrangian Heterogeneous Multiscale Method (LHMM) models polymeric melts and solutions. It describes the macrocopic scales using a GENERIC-compliant SPH discretization, whereas at microscales it adopt the Dissipative Particle Dynamics method.
    

## The Code
 LHMM is currently implemented using a C++ driver, that uses LAMMPS as a library to concurrently run macro and micro simulations. GPU accelerated version of DPD is used to run microsimulations.

## Acknowledgments 

LHMM-PolymerDPD is being developed at The Basque Center for Applied Mathematics. The authors acknowledge the funding provided by IKUR–HPC&AI – (HPCAI10: MOLD-POLY) and the IKUR Strategy funded by  Basque Government and the European Union NextGenerationEU/PRTR. The research is also partially funded by the Spanish State Research Agency through BCAM Severo Ochoa excellence accreditation CEX2021-0011 42-S/MICIN/AEI/10.13039/501100011033, and through the project PID2024-158994OB-C42 (‘Multiscale Modeling of Friction, Lubrication, and Viscoelasticity in Particle Suspensions’ and acronym ‘MMFLVPS’) funded by MICIU/AEI/10.13039/501100011033 and cofunded by the European Union. The authors thankfully acknowledges the computer resources at MareNostrum and the technical support provided by Barcelona Supercomputing Center (IM-2024-3-0013 and IM-2025-2-0042)

