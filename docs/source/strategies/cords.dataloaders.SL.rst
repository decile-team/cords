Supervised Learning Subset Selection Data Loaders
==================================================
In this section, we consider different subset selection based data loaders geared towards efficient and robust learning 
in standard supervised learning setting.

DSS Dataloader (Base Class)
-----------------------------------------------

.. automodule:: cords.utils.data.dataloader.SL.dssdataloader
   :members:
   :undoc-members:
   :show-inheritance:

Non-Adaptive subset selection Data Loaders
-------------------------------------------
    Non Adaptive DSS Dataloader (Base Class of Non-Adaptive dataloaders)
    ---------------------------------------------------------------------    
    .. automodule:: cords.utils.data.dataloader.SL.nonadaptive.nonadaptivedataloader
        :members:
        :undoc-members:
        :show-inheritance:

    Non Adaptive CRAIG Dataloader :footcite:`pmlr-v119-mirzasoleiman20a`
    ----------------------------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.nonadaptive.craigdataloader
        :members:
        :undoc-members:
        :show-inheritance:
   
    Non Adaptive Submodular Dataloader
    -------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.nonadaptive.submoddataloader
        :members:
        :undoc-members:
        :show-inheritance:

Adaptive subset selection Data Loaders
-------------------------------------------
    Adaptive DSS Dataloader (Base Class of Adaptive dataloaders)
    ---------------------------------------------------------------------    
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.adaptivedataloader
        :members:
        :undoc-members:
        :show-inheritance:

    GLISTER Dataloader:footcite:`killamsetty2021glister`
    -----------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.glisterdataloader
        :members:
        :undoc-members:
        :show-inheritance:

    CRAIG Dataloader:footcite:`pmlr-v119-mirzasoleiman20a`
    -------------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.craigdataloader
        :members:
        :undoc-members:
        :show-inheritance:

    GradMatch Dataloader:footcite:`pmlr-v139-killamsetty21a`
    ------------------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.gradmatchdataloader
        :members:
        :undoc-members:
        :show-inheritance:

    Random Dataloader
    ---------------------------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.randomdataloader
        :members:
        :undoc-members:
        :show-inheritance:

    Random-Online Dataloader
    ---------------------------------------------------------------------
    .. automodule:: cords.utils.data.dataloader.SL.adaptive.olrandomdataloader
        :members:
        :undoc-members:
        :show-inheritance:

REFERENCES
-----------
.. footbibliography::
