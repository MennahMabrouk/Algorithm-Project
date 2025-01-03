## Algorithm-Project

- We will run algorithms for sequence alignment and then compare between them:
  - **Smith-Waterman**
  - **Particle Swarm Optimization**
  - **Sine Cosine Algorithm**
  - **SCA-PSO Alignment**

---

## **Algorithms and Updating Equations**

### **1. Smith-Waterman Algorithm**

The Smith-Waterman algorithm is used for local sequence alignment to identify regions of similarity between sequences.

**Score Update Equation:**

$$
\text{Score}(i, j) = \max \begin{cases} 
\text{Score}(i-1, j-1) + \text{Similarity}(\text{SeqA}[i], \text{SeqB}[j]) \\
\text{Score}(i-1, j) + \text{GapPenalty} \\
\text{Score}(i, j-1) + \text{GapPenalty} \\
0 
\end{cases}
$$

Where:

- \( \text{Similarity}(\text{SeqA}[i], \text{SeqB}[j]) \): A positive score for a match and a negative score for a mismatch.  
- \( \text{GapPenalty} \): A penalty for introducing gaps in the alignment.

---

### **2. Particle Swarm Optimization (PSO)**

PSO is a metaheuristic optimization algorithm inspired by the social behavior of birds or fish. Each particle in the population adjusts its position based on its own experience and the global best position.

**Velocity Update Equation:**

$$
v_i^{t+1} = w \cdot v_i^t + c_1 \cdot \text{rand}() \cdot (p_{i,\text{best}} - x_i^t) + c_2 \cdot \text{rand}() \cdot (g_{\text{best}} - x_i^t)
$$

**Position Update Equation:**

$$
x_i^{t+1} = x_i^t + v_i^{t+1}
$$

Where:

- \( w \): Inertia weight controlling the influence of the previous velocity.  
- \( c_1, c_2 \): Cognitive and social coefficients.  
- \( \text{rand}() \): Random value between \( [0, 1] \).  
- \( p_{i,\text{best}} \): Best position of particle \( i \).  
- \( g_{\text{best}} \): Global best position.

---

### **3. Sine-Cosine Algorithm (SCA)**

SCA uses sine and cosine functions to explore and exploit the search space dynamically.

**Position Update Equation:**

$$
x_i^{t+1} =
\begin{cases}
x_i^t + r_1 \cdot \sin(r_2) \cdot |r_3 \cdot g_{\text{best}} - x_i^t|, & \text{if } r_4 < 0.5 \\
x_i^t + r_1 \cdot \cos(r_2) \cdot |r_3 \cdot g_{\text{best}} - x_i^t|, & \text{otherwise.}
\end{cases}
$$

Where:

- \( r_1 \): Exploration factor controlling the step size, reduced over iterations.  
- \( r_2 \): A random angle in \( [0, 2\pi] \) controlling the direction of movement.  
- \( r_3, r_4 \): Random values in \( [0, 1] \).  
- \( g_{\text{best}} \): Global best solution.

**Exploration Factor:**

$$
r_1 = a \cdot \left( 1 - \frac{t}{T} \right)
$$

Where:

- \( a \): A constant factor.  
- \( t \): Current iteration.  
- \( T \): Maximum number of iterations.

---

### **4. SCA-PSO Algorithm**

SCA-PSO combines the Sine-Cosine Algorithm (SCA) and Particle Swarm Optimization (PSO) to balance exploration and exploitation.

#### **Bottom Layer (SCA)**

**Position Update Equation:**

$$
x_{ij}^{t+1} =
\begin{cases}
x_{ij}^t + r_1 \cdot \sin(r_2) \cdot |r_3 \cdot y_i - x_{ij}^t|, & \text{if } r_4 < 0.5 \\
x_{ij}^t + r_1 \cdot \cos(r_2) \cdot |r_3 \cdot y_i - x_{ij}^t|, & \text{otherwise.}
\end{cases}
$$

#### **Top Layer (PSO)**

**Velocity Update Equation:**

$$
v_i^{t+1} = w \cdot v_i^t + c_1 \cdot \text{rand}() \cdot (y_{i,\text{pbest}} - y_i^t) + c_2 \cdot \text{rand}() \cdot (y_{\text{gbest}} - y_i^t)
$$

**Position Update Equation:**

$$
y_i^{t+1} = y_i^t + v_i^{t+1}
$$

Where:

- \( y_i \): Best solution of group \( i \) in the bottom layer.  
- \( y_{i,\text{pbest}} \): Personal best solution of \( y_i \).  
- \( y_{\text{gbest}} \): Global best solution across all particles.

---

### **5. FLAT Algorithm**

The **Fragmented Local Alignment Technique (FLAT)** is designed to detect the **Longest Common Consecutive Subsequence (LCCS)** between two sequences. It works by fragmenting sequences into smaller subsequences of size \( L \) and performing local alignment using techniques like the **Smith-Waterman** algorithm.

At each iteration, FLAT compares all possible fragments of length \( L \) in two sequences \( \text{SeqA} \) and \( \text{SeqB} \):

$$
\text{Score}(i, j) = \max_{1 \leq k \leq L} \left( \text{Smith-Waterman}(\text{SeqA}[i:i+L], \text{SeqB}[j:j+L]) \right)
$$

Where:

- \( \text{SeqA}[i:i+L] \): Fragment of length \( L \) starting at position \( i \) in sequence \( \text{SeqA} \).  
- \( \text{SeqB}[j:j+L] \): Fragment of length \( L \) starting at position \( j \) in sequence \( \text{SeqB} \).  
- \( \text{Smith-Waterman}(x, y) \): Local alignment score between fragments \( x \) and \( y \).

The FLAT algorithm iterates over all possible starting positions \( i \) and \( j \) in the sequences:

$$
\text{Best Score} = \max_{i, j} \left( \text{Score}(i, j) \right)
$$

Where:

- \( i \in [0, |\text{SeqA}| - L] \)  
- \( j \in [0, |\text{SeqB}| - L] \)

---

## **Dependencies**

You can install the required libraries using `pip`:

```bash
pip install numpy biopython matplotlib pyswarm scipy fpdf pandas
