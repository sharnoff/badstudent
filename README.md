# badstudent

This is currently in progress -- there will be large API changes in the near future.
Once the API is stable, documentation will be added outside of the codebase.

Current features:
* different types of operations available as layers
* free-form layer architecture (no recurrent features yet)
* custom cost functions for network output
* custom optimization functions for correction
* hassle-free supplying of data to train the network
* real-time output of training status
* custom run conditions
* custom functions for variable learning rates

Future features:
* saving / loading of networks

Will be testing using cmd/xor, which just checks that the network can learn to be an XOR gate, and cmd/mnist