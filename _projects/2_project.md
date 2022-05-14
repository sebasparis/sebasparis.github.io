---
layout: post
title: Floquet-Born-Markov C++ libraries
description: Free-to-use libraries to solve periodically driven open quantum system dynamics.
---

# Floquet-Born-Markov C++ libraries

This public resository contains two libraries that help solve open and closed quantum system dynamics.

The library floquet.h lets calculate several steady state parameters and explicit time evolution for an arbitrary quantum system with driven Hamiltonian of the form $H(t) = V + A\cdot f(t)$, with $f(t)$ a function periodic in time.

The library FBM_mRWA also lets calculate steady state parameters and explicit time evolution but for the case of an open driven quantum system, weakly coupled to one or several thermal baths so that the Born an Markov approximations can be appilied. An additional RWA is also applied, also related to weak thermal bath coupling with respect to the driving frequency.

Github link: https://github.com/sebasparis/FBM_mRWA