# smartlearning

This is currently in progress -- there will be large API changes in the near future.
Once the API is stable, documentation will be added outside of the codebase.

Current features:
* layers of feed-forward neurons with the logistic function serving as the activation funtion
* easy supplying of data to train the network
* real-time output of training status
* custom cost functions for network output
* free-form layer architecture (no recurrent features yet)
* custom optimization functions for correction

Future features:
* different types of operations available as layers
* custom functions for variable learning rates
* saving / loading of networks

Will be testing using cmd/test, which just checks that the network can learn to be an XOR gate