### CORDS has incorporated popular model architectures like DenseNet, ResNet, MobileNet, VGG etc

### To integrate any new custom model architecture, specific alterations are required in the forward functions of the model architecture implementations. Specifically, the forward method should contain two additional variables:

1. A Boolean variable 'last' that, if set to true, returns both the model output and the output of the penultimate layer. If set to false, it merely returns the model output.

2. A Boolean variable 'freeze' that, if set to true, prevents the construction of a computational graph for all computations prior to the final fully connected layer. Essentially, gradient calculation over the model parameters prior to the final fully connected layer is omitted. Conversely, if set to false, it constructs a computational graph for all computations within the model.

Moreover, one needs to introduce a new function called get\_embedding\_dim() method, which returns the dimensions of the input to the final fully connected layer. 
ayer.
