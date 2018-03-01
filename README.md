# arxivTitleGenerator_RNN
Trains an RNN on paper titles from the arXiv and generates new titles.

The code for this project was based on work for the Shakespeare generator that can be found at 
https://github.com/martin-gorner/tensorflow-rnn-shakespeare, credit goes to Martin Gorner for the structure of the network

I mostly did this so I would have an excuse to learn more about Tensorflow, and in that regard it has been very effective. The LSTM version has a minor bug involving using a tuple for the states at each layer, but the problem has been solved and just needs a little testing.

Results:
The generator works better than I might have expected. It's pretty clear that there are some topics that are quite popular (especially titles with "stability" or "spectral" since that shows up a lot in the generated text). There are a lot of rare words in the titles, such as specific names of theorems (ex: "Karpenko-Merkurjev theorem" appears exactly once in the two years of titles from Jan 2016 to Dec 2017) and these are not learned. Still, there are some plausible sounding titles being generated, and proper LaTeX was learned.


A sample of "good" titles:
"A class of non-commutative representation of singular sets"
"A class of nonlinear schr\""odinger operators"
"A new construction of the singular propagation of the complex structure of a closed operator"
"Singularities and structure of two-space simplicial groups"
"The stability of solutions of a class of nonlinear equations with singular potentials"
"Topology and applications of the contractive space of a complete graph and the structure of three point spaces"
"The cohomology of the polyhedral group"



A sample of "bad" titles:
"The complex structure of statistical polynomial of the classical and subspace of space of surfaces"
"The constant maps in the semi-classical stability of the complete"
"Applications to a conservative programming on the space of the space of a finite field of applications"
"A new proof of the complete continuous functional of the completely semidefinite probability"
"A conjecture of the sphere of a complete graph of a conjecture spaces and their applications to the spatial distribution of an algebraic structure of the complex"
"Spectral stability of the spectrum of the space of the classification of some complex surfaces in $\mathbb{R}^n$"
"An algebra on the spectral spectrum of the singulir polynomial of a complete group"
