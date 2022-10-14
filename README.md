# Ensuring thermodynamic consistency with invertible coarse-graining


Official implementation of:  

**Ensuring thermodynamic consistency with invertible coarse-graining**

Shriram Chennakesavalu, David J. Toomer and Grant M. Rotskoff



**Abstract**: Coarse-grained models are a core computational tool in theoretical chemistry
and biophysics. A judicious choice of a coarse-grained model can yield physical
insight by isolating the essential degrees of freedom that dictate the
thermodynamic properties of a complex, condensed-phase system. The reduced
complexity of the model typically leads to lower computational costs and more
efficient sampling compared to atomistic models. Designing "good"
coarse-grained models is an art. Generally, the mapping from fine-grained
configurations to coarse-grained configurations itself is not optimized in any
way; instead, the energy function associated with the mapped configurations is.
In this work, we explore the consequences of optimizing the coarse-grained
representation alongside its potential energy function. We use a graph machine
learning framework to embed atomic configurations into a low dimensional space
to produce efficient representations of the original molecular system. Because
the representation we obtain is no longer directly interpretable as a real
space representation of the atomic coordinates, we also introduce an inversion
process and an associated thermodynamic consistency relation that allows us to
rigorously sample fine-grained configurations conditioned on the coarse-grained
sampling. We show that this technique is robust, recovering the first two
moments of the distribution of several observables in proteins such as
chignolin and alanine dipeptide.

