Subset Selection Dataloaders
==============================
Essentially, with subset selection-based data loaders, it is pretty straightforward to use subset selection strategies directly 
because they are integrated directly into subset data loaders; this allows users to use subset selection strategies directly by 
using their respective subset selection data loaders.

Below is an example that shows the subset selection process is simplified by just calling a data loader in supervised learning setting,

.. code-block:: python
    :emphasize-lines: 23-31

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
            Standard PyTorch training loop using weighted loss
            
            Our training loop differs from the standard PyTorch training loop in that along with 
            data samples and their associated target labels; we also have additional sample weight
            information from the subset data loader, which can be used to calculate the weighted 
            loss for gradient descent. We can calculate the weighted loss by using default PyTorch
            loss functions with no reduction as follows:       
            """
            # Convert inputs, targets, and weights to the required device
            inputs = inputs.to(self.cfg.train_args.device)
            targets = targets.to(self.cfg.train_args.device, non_blocking=True)
            weights = weights.to(self.cfg.train_args.device)
            
            # Zero the optimizer gradients to prevent gradient accumulation
            optimizer.zero_grad()

            #Model forward pass over the inputs
            outputs = model(inputs)

            # Get individual sample losses with no reduction 
            losses = criterion_nored(outputs, targets) 

            # Get weighted loss by a dotproduct of the losses vector with sample weights
            loss = torch.dot(losses, weights / (weights.sum())) 

            # Do backprop on the weighted loss
            loss.backward() 
            
            # Step the model based on the gradient values
            optimizer.step()

In our current version, we deployed subset selection data loaders in supervised learning and semi-supervised learning settings.

.. toctree::

   cords.dataloaders.SL
   cords.dataloaders.SSL
