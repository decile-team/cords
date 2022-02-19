Subset Selection Dataloaders
==============================
Essentially, with subset selection-based data loaders, it is pretty straightforward to use subset selection strategies directly 
because they are integrated directly into subset data loaders; this allows users to use subset selection strategies directly by 
using their respective subset selection data loaders.

Below is an example that shows the subset selection process is simplified by just calling a data loader in supervised learning setting,

.. code-block:: python
    
    dss_args = dict(model=model,
                    loss=criterion_nored,
                    eta=0.01,
                    num_classes=10,
                    num_epochs=300,
                    device='cuda',
                    fraction=0.1,
                    select_every=20,
                    kappa=0,
                    linear_layer=False,
                    selection_type='SL',
                    greedy='Stochastic')
    dss_args = DotMap(dss_args)

    dataloader = GLISTERDataLoader(trainloader, valloader, dss_args, logger, 
                                    batch_size=20, 
                                    shuffle=True,
                                    pin_memory=False)
    
    for epoch in range(num_epochs):
        for _, (inputs, targets, weights) in enumerate(dataloader):
            """
            Standard PyTorch Training Loop
            """

In our current version, we deployed subset selection data loaders in supervised learning and semi-supervised learning settings.

.. toctree::

   cords.dataloaders.SL
   cords.dataloaders.SSL
