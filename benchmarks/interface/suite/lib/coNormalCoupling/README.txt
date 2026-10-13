libcoNormalCoupling -- user library for OpenFOAM v2606 (ESI), built in this folder only.

Build (nothing is written into the OpenFOAM installation):
    nice -n 10 openfoam2606 -c ./Allwmake
The target is the LIB line of Make/files (./lib, relative to this folder).

Contents
    coNormalTemperatureCoupledMixedFvPatchScalarField.{H,C}
        boundary condition  compressible::coNormalTemperatureCoupledMixed  (both sides of the interface):
        stock coupled condition + tangential part of the co-normal flux + isotropy guard
    anisotropicCouplingGuard.{H,C}
        function object: stops (or warns about) kappaMethod solidThermo on an anisotropic solid, for cases
        that keep the stock boundary condition
    coNormalWallHeatFlux.{H,C}
        function object: wall heat flux of a solid region with the tensor operator the solver uses
    coNormalFlux.{H,C}
        the two flux expressions shared by the above

Use: add   libs ("<this folder>/lib/libcoNormalCoupling.so");   to system/controlDict; usage blocks are in the
header of each class.  What was tested, and the conditional stability of the boundary condition at high
anisotropy ratio, are described in the article and its Supplementary Material.
