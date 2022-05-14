---
layout: post
title: Quantum Approximate Optimization Algorithm
description: Implementation of a quantum agorithm that solves the Max-Cut problem for a random graph.
---
# Algorithm overview
The Quantum Approximate Optimization Algorithm (QAOA) is a quantum algorithm introduced by Edward Fahri, Jeffrey Goldstone and Sam Gutmann [[1]](https://arxiv.org/abs/1411.4028)  that finds approximate solutions of a classical optimization problem over a given set of bitstrings.

The general idea of the algorithm is to produce a quantum state which, after measurement in the computational basis, collapses with high probability into bitstrings that are at or close to the optimum value.

In the language of the original paper, one can define the objective function to be maximized as a sum over clauses $\alpha$:

$$
C(z) = \sum_{\alpha=1}^m C_\alpha(z)
$$

In the last expression, $z=z_1 z_2 ... z_n$ is the bitsring and $C_\alpha = 1$ if $z$ satisfies the clause $\alpha$ and $0$ otherwise. Typically, $C_\alpha(z)$ only involves as few of the $n$ bits. The optimization problem consists in finding $z$ such that $C(z)$ is as great as possible.

The problem that is asked to be solved is the MaxCut problem over a graph $G$. The MaxCut problem consists on finding two complementary subsets of the vertices of $G$ that maximize the number of edges connecting the two subsets [[2]](https://en.wikipedia.org/wiki/Maximum_cut). 

Every position along the bitstring $z$ represents a certain vertex of $G$. The value $0$ or $1$ in that position represents whether the vertex belongs to one or another subset of the paritition of $G$ defined by $z$. $C(z)$ is the number of shared edges between the two subsets of the partition.

For this problem we can map $C(z)$ to an Ising-like Hamiltonian [[3]](https://internal-journal.frontiersin.org/articles/10.3389/fphy.2014.00005/full):

$$
C(z) = \sum_{\langle jk \rangle} C_{\langle j k \rangle},
$$

where $\langle j k \rangle$ runs over all the edges of $G$ connecting the vertices $j$ and $k$, and

$$
C_{\langle j k \rangle} = \frac{1}{2}(1-\sigma_z^{j} \sigma_z^k),
$$

with $\sigma_i^j$ the $i$th Pauli matrix acting on the $j$th qubit. Defining the operator $B$ as

$$
B = \sum_{j=1}^n \sigma_x^j,
$$

and the unitary operator $U(A,\theta)$ that applies a rotation of angle $\theta$ along the operator $A$,

$$
U(A,\theta) = e^{-i\theta A}
$$

the QAOA consists on, for a given integer $p$ which is related to the precision of the algorithm, finding $2p$ angles $\gamma = \gamma_1 ... \gamma_p$ and $\beta=\beta_1 ... \beta_p$ such that the quantum state

$$
|\mathbf{\gamma\beta}\rangle = U(B,\beta_p)U(C,\gamma_p)...U(B,\beta_1)U(C,\gamma_1)|s\rangle
$$

maximizes the expectation value $\langle \mathbf{\gamma\beta}|C|\mathbf{\gamma\beta}\rangle$, with $|s\rangle$ the uniform superposition of all computational basis states
$$
|s\rangle = \frac{1}{2^{n/2}}\sum_z |z\rangle
$$
Then, one performs a measurement over this state, which has a high probability of yielding a bitstring with a high value of $C(z)$. 

In some special cases, an expression of the optimal angles $\mathbf{\gamma}$ and $\mathbf{\beta}$ for a given $p$ can be obtained analytically or calculated a priori with a classical computer. I am not aware of such a result for a random graph. Therefore in this case we make use of a classical optimization algorithm to maximize the expectation value of $C$.

# Implementation

In this section the QAOA is implemented using qiskit.

First we will construct a random graph of $n$ vertices. A random graph over $n$ vertices can be built by considering all possible pairs of vertices, and joining them with an edge with a probability $\rho$.


```python
n = 12 #Number of vertices/qubits
rho = 3/12 #Probability that two vertices share an edge
```

Now we create this random graph with networkx.



```python
import networkx as nx
G = nx.fast_gnp_random_graph(n,rho)
```

Our graph looks like this:


```python
nx.draw(G,with_labels=True)
```


    
![png](assets/images/output_7_0.png)
    


Now that we have our graph $G$, we focus on implementing the algorithm. Beforehand we will define its precision $p$:


```python
p=2
```

The algorithm will be performed on a quantum register of $n$ qubits, with $n$ classical bits for measurement. The initial state of the circuit can be set to $|s\rangle$ by initializing the circuit with all qubits in the state $|0\rangle$, and then applying a Hadamard gate in every qubit. Implementing this as a quantum circuit and naming it <i>init_circ</i>,


```python
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
init_circ = QuantumCircuit(n)
for i in range(n):
    init_circ.h(i)
```

This circuit looks like this:


```python
init_circ.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">      ┌───┐
 q_0: ┤ H ├
      ├───┤
 q_1: ┤ H ├
      ├───┤
 q_2: ┤ H ├
      ├───┤
 q_3: ┤ H ├
      ├───┤
 q_4: ┤ H ├
      ├───┤
 q_5: ┤ H ├
      ├───┤
 q_6: ┤ H ├
      ├───┤
 q_7: ┤ H ├
      ├───┤
 q_8: ┤ H ├
      ├───┤
 q_9: ┤ H ├
      ├───┤
q_10: ┤ H ├
      ├───┤
q_11: ┤ H ├
      └───┘</pre>



Next we create quantum circuits that implement the operations $U(C,\gamma)$ and $U(B,\beta)$ for arbitrary angles. 

Let us start by creating a function that returns an $n$-qubit quantum circuit that implements the operation $U(B,\beta)$, given the angle $\beta$. For this we implement a $RX(2\beta)$ gate over each qubit. The factor of two appears because

$$
RX(2\beta) = e^{-i\beta\sigma_x}
$$

which is the operation we need to perform $U(B,\beta)$.


```python
#Define the function that returns the circuit U(B,β)
def UB(beta):
    UB_circ = QuantumCircuit(n)
    for i in range(n):
        UB_circ.rx(2*beta,i)
    return UB_circ
```

The circuit that implements $U(B,\beta)$ looks like this:


```python
#Draw the circuit that implements U(B,β) for a parameter β
from qiskit.circuit import Parameter
beta_par =  Parameter("β")
UB(beta_par).draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">      ┌─────────┐
 q_0: ┤ Rx(2*β) ├
      ├─────────┤
 q_1: ┤ Rx(2*β) ├
      ├─────────┤
 q_2: ┤ Rx(2*β) ├
      ├─────────┤
 q_3: ┤ Rx(2*β) ├
      ├─────────┤
 q_4: ┤ Rx(2*β) ├
      ├─────────┤
 q_5: ┤ Rx(2*β) ├
      ├─────────┤
 q_6: ┤ Rx(2*β) ├
      ├─────────┤
 q_7: ┤ Rx(2*β) ├
      ├─────────┤
 q_8: ┤ Rx(2*β) ├
      ├─────────┤
 q_9: ┤ Rx(2*β) ├
      ├─────────┤
q_10: ┤ Rx(2*β) ├
      ├─────────┤
q_11: ┤ Rx(2*β) ├
      └─────────┘</pre>



Now we implement the function that returns circuit that implements the $U(C,\gamma)$ operation given $\gamma$. For this, we implement a two-qubit RZZ(-$\gamma$) gate, since for two qubits $i$ and $j$
$$
RZZ_{ij}(-\gamma) = e^{i\frac{\gamma}{2}\sigma_z^i \sigma_z^j} \propto U(C_{\langle ij\rangle},\gamma)
$$
where the proportionality is up to an irrelevant global phase. Note that since the $C_{\langle ij \rangle}$ are all diagonal in the computational basis, they all commute with each other and therefore
$$
U(C,\gamma) = U(\sum_{\langle ij \rangle}C_{\langle ij \rangle},\gamma) = \prod_{\langle ij \rangle} U(C_{\langle ij \rangle},\gamma)
$$
where the last product can be taken in any order. This means that the RZZ gates of the circuit can be performed in any order.


```python
def UC(gamma):
    UC_circ = QuantumCircuit(n)
    for edge in list(G.edges()):
        UC_circ.rzz(-gamma,edge[0],edge[1])
    return UC_circ
```

The circuit that implements $U(C,\gamma)$ looks like this for our graph $G$:


```python
gamma_par = Parameter('γ')
UC(gamma_par).draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">                                                                       »
 q_0: ─■────────────■────────────■─────────────────────────■───────────»
       │ZZ(-1.0*γ)  │            │                         │           »
 q_1: ─■────────────┼────────────┼────────────■────────────┼───────────»
                    │ZZ(-1.0*γ)  │            │ZZ(-1.0*γ)  │           »
 q_2: ──────────────■────────────┼────────────■────────────┼───────────»
                                 │                         │           »
 q_3: ─■─────────────────────────┼────────────■────────────┼───────────»
       │                         │            │            │           »
 q_4: ─┼─────────────────────────┼────────────┼────────────┼───────────»
       │ZZ(-1.0*γ)               │            │            │           »
 q_5: ─■─────────────────────────┼────────────┼────────────┼───────────»
                                 │            │            │           »
 q_6: ───────────────────────────┼────────────┼────────────┼───────────»
                                 │ZZ(-1.0*γ)  │ZZ(-1.0*γ)  │           »
 q_7: ───────────────────────────■────────────■────────────┼───────────»
                                                           │           »
 q_8: ─────────────────────────────────────────────────────┼───────────»
                                                           │ZZ(-1.0*γ) »
 q_9: ─────────────────────────────────────────────────────■───────────»
                                                                       »
q_10: ─────────────────────────────────────────────────────────────────»
                                                                       »
q_11: ─────────────────────────────────────────────────────────────────»
                                                                       »
«                                                                       »
« q_0: ──────────────■──────────────────────────────────────────────────»
«                    │                                                  »
« q_1: ──────────────┼────────────■─────────────────────────────────────»
«                    │            │                                     »
« q_2: ─■────────────┼────────────┼────────────■────────────────────────»
«       │            │            │            │                        »
« q_3: ─┼────────────┼────────────┼────────────┼────────────────────────»
«       │ZZ(-1.0*γ)  │            │            │                        »
« q_4: ─■────────────┼────────────┼────────────┼────────────■───────────»
«                    │            │            │            │ZZ(-1.0*γ) »
« q_5: ──────────────┼────────────┼────────────┼────────────■───────────»
«                    │            │            │                        »
« q_6: ──────────────┼────────────┼────────────┼────────────────────────»
«                    │            │            │                        »
« q_7: ──────────────┼────────────┼────────────┼────────────────────────»
«                    │            │            │                        »
« q_8: ──────────────┼────────────┼────────────┼────────────────────────»
«                    │            │ZZ(-1.0*γ)  │                        »
« q_9: ──────────────┼────────────■────────────┼────────────────────────»
«                    │                         │ZZ(-1.0*γ)              »
«q_10: ──────────────┼─────────────────────────■────────────────────────»
«                    │ZZ(-1.0*γ)                                        »
«q_11: ──────────────■──────────────────────────────────────────────────»
«                                                                       »
«                                                                       »
« q_0: ─────────────────────────────────────────────────────────────────»
«                                                                       »
« q_1: ─────────────────────────────────────────────────────────────────»
«                                                                       »
« q_2: ─────────────────────────────────────────────────────────────────»
«                                                                       »
« q_3: ─────────────────────────────────────────────────────────────────»
«                                                                       »
« q_4: ─■───────────────────────────────────────────────────────────────»
«       │                                                               »
« q_5: ─┼────────────■────────────■─────────────────────────────────────»
«       │ZZ(-1.0*γ)  │            │                                     »
« q_6: ─■────────────┼────────────┼────────────■────────────────────────»
«                    │ZZ(-1.0*γ)  │            │                        »
« q_7: ──────────────■────────────┼────────────┼────────────────────────»
«                                 │            │ZZ(-1.0*γ)              »
« q_8: ───────────────────────────┼────────────■────────────■───────────»
«                                 │ZZ(-1.0*γ)               │           »
« q_9: ───────────────────────────■────────────■────────────┼───────────»
«                                              │            │ZZ(-1.0*γ) »
«q_10: ────────────────────────────────────────┼────────────■───────────»
«                                              │ZZ(-1.0*γ)              »
«q_11: ────────────────────────────────────────■────────────────────────»
«                                                                       »
«                   
« q_0: ─────────────
«                   
« q_1: ─────────────
«                   
« q_2: ─────────────
«                   
« q_3: ─────────────
«                   
« q_4: ─────────────
«                   
« q_5: ─────────────
«                   
« q_6: ─────────────
«                   
« q_7: ─────────────
«                   
« q_8: ─────────────
«                   
« q_9: ─────────────
«                   
«q_10: ─■───────────
«       │ZZ(-1.0*γ) 
«q_11: ─■───────────
«                   </pre>



Finally, we put all the pieces together, and build a function that, given vectors $\gamma=(\gamma_1 ... \gamma_p)$ and $\beta=(\beta_1 ... \beta_p)$, returns a quantum circuit that implements the QAOA, including measurement of the qubits. We do this by properly concatenating the circuits that we just created.


```python
def qaoa(gamma,beta):
    circ = QuantumCircuit(n)
    circ.append(init_circ,[i for i in range(n)])
    for k in range(p):
        circ.append(UC(gamma[k]),[i for i in range(n)])
        circ.append(UB(beta[k]),[i for i in range(n)])
    circ.measure_all()
    return circ.decompose()
```

Now, using the function above, we will construct another function that recieves the vectors $\mathbf{\gamma}$ and $\mathbf{\beta}$ as input, and returns the expected value of $C$ for the circuit output. Then we will maximize this function using a classical optimizer.

We will first define a function shared_edges($G$,$z$), that for a given partition of $G$ defined by the bitstring $z$, returns the number of shared edges between the two partitions. 


```python
def shared_edges(G,z):
    count = 0
    for i, j in G.edges():
        if z[i] != z[j]:
            count += 1
    return count
```

Now we define the function to be optimized, that returns the expectation value of $C$, given as input a vector $\theta=[\gamma,\beta]$ of length $2 p$, whose first $p$ coordinates are the coordinates of $\gamma$ and its next $p$ coordinates are the coordinates of $\beta$.


```python
from qiskit import Aer, execute
p=2
def C_exp_value(theta):
    gamma = theta[:p]
    beta = theta[p:]
    
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2**16
    circ = qaoa(gamma,beta)
    counts = backend.run(circ,nshots=2**16).result().get_counts()
    
    sum_maxcut_edges = 0
    sum_counts = 0
    
    for z, count in counts.items():
        sum_maxcut_edges += count*shared_edges(G,z)
        sum_counts += count
    
    return sum_maxcut_edges/sum_counts
```

Finally, use a classical optimizer to find the maximum of C_exp_value (for this we actually minimze -C_exp_value)


```python
from scipy.optimize import minimize
def minus_C_exp_value(theta):
    return -C_exp_value(theta)
res = minimize(minus_C_exp_value, [1.,1.,1.,1.], method='COBYLA')
print(res)
```

         fun: -9.7265625
       maxcv: 0.0
     message: 'Optimization terminated successfully.'
        nfev: 49
      status: 1
     success: True
           x: array([1.6670911 , 2.47768103, 1.23475326, 2.06188397])


Now that we have found the optimal angles, let's run the circuit with these angles, and make a batch of measurements over the resulting state.


```python
circ_res = qaoa(res.x[:p],res.x[p:])
backend = Aer.get_backend('qasm_simulator')
backend.shots = 2**16
counts = backend.run(circ_res, shots = 2**16).result().get_counts()
```

Our solution is the bitstring with the highest counts in this batch of measurements.


```python
z_solution = max(counts, key=counts.get)
```


```python
print('The bitstring with the highest counts ('+ str(counts[z_solution]) + ') is ' + z_solution + '.')
print('It has C=' + str(shared_edges(G,z_solution)) +' shared edges between subsets.')
```

    The bitstring with the highest counts (220) is 100110100001.
    It has C=11 shared edges between subsets.


Let's make a color map for our graph using our solution. We will color vertices marked as $0$ with blue, and vertices marked as $1$ with red.


```python
color_map = []
for node in G:
    if z_solution[node] == '0':
        color_map.append('blue')
    else: 
        color_map.append('red')
```

The solution found by the QAOA looks like this:


```python
nx.draw(G, node_color=color_map, with_labels=True)
```


    
![png](assets/images/output_38_0.png)
