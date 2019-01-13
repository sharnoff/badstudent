[![GoDoc](https://godoc.org/github.com/sharnoff/badstudent?status.png)](http://godoc.org/github.com/sharnoff/badstudent)

# Badstudent

### What is it?

badstudent is a general-purpose machine-learning package written in and for go. It's not designed to be super
effective - I'm just using it as a way to learn more about the lower-level parts of neural networks. It
excels at providing flexibility to the user - including multiple-time-step delay, free-form architecture, and
easily definable layers.

This framework has been a solo project, but it's perfectly usable for other people. (There are better
libraries out there, though)

### How do I use it?

badstudent is based on Nodes and Operators. Nodes are essentially the wrappers around the 'user supplied'
Operators. The Operators are only practically supplied by other parts of the package (badstudent/operators).
Some of these Operators (such as a layer of neurons) require additional Optimizers, which also can either be
supplied by the user or the package (badstudent/operators/optimizers).

There is a ton of available information available on godoc about this package.