# robust-fl-reweighing

### misc.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

### files
- fl_main.py: main FL pipeline
- models.py: contains all model definitions (classes)
- aggregation.py: contains all aggregation algorithms
- utils.py: functions for training and testing

### to-do
- Look into normalization for data loaders
- Clean up imports in the various files
- Maybe change up the optimizer

### questions to ask
- Should I be shuffling test data in the loader or no?
- What does num clients mean for the loader, should I use it?
- Is it plagarism if I copy code from my scribe example?