# robust-fl-reweighing

### misc.
CIFAR-10 classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

### files
- fl_main.py: main FL pipeline
- models.py: contains all model definitions (classes)
- aggregation.py: contains all aggregation algorithms
- utils.py: functions for training and testing

### to-do
- Maybe change accuracy to f1
- Look into normalization for data loaders
- Clean up imports in the various files
- Maybe change up the optimizer
- What does num clients mean for the loader, should I use it?