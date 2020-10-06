# Helmholtz

Coil Musings
============

Numerical study of uniformity in the central region of a pair of Helmholtz 
coils as the coil size and spacing is varied.

The analyses here are done using David Meeker's FEMME finite element package
and feature a pair of 1m diameter coils lying parallel to the xy plane and
generating field along the z axis.
Various configurations are considered.

**PureHelmholtz** uses point coils spaced exactly 50cm apart in the center of a 2m
diameter sphere with infinite boundary conditions. It detected a *tiny* asymmetry
about the z=0 plane.

**PureHelmholtzI** replaced the inifinite boundary condition with default boundary 
conditions, which pulled the field into the walls. The asymmetry is unaltered.

**PureHelmholtzHR** kept the infinite bounds but significantly increased the
resolution in the central region. The asymmetry is unaltered.



