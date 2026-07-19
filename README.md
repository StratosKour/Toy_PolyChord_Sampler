# Toy_PolyChord_Sampler
An educational, Nested Sampling algorithm in Python for D dimensions (We used 5-dimensions). Inspired by PolyChord, it features Slice Sampling and adaptive covariance to calculate Bayesian Evidence. Includes visual tracing and anesthetic GUI support.

# How It Works
The algorithm is designed to map complex probability distributions by searching for the peak likelihood region. Its engine relies on two core mechanics:

1. Slice Sampling (Neal's Algorithm): To find new, higher-likelihood replacements for "dead" points, the code doesn't guess blindly. It uses a "Stepping-out & Shrinkage" slice sampling method, casting a line segment in a random direction and shrinking it until a valid point is found.
2. Adaptive Exploration (Covariance Updating): The algorithm learns the shape of the probability "mountain" as it climbs. By keeping track of intermediate "phantom points" generated during the MCMC random walks, it continuously updates its covariance matrix to adapt its search compass to the narrowing parameter space.

# Toy problem
To test the algorithm's accuracy, it is given a hidden target: a 5D Multivariate Normal (Gaussian) distribution strictly centered at (0.5*D), bounded by a uniform prior from 0 to 1.

The sampler starts by scattering points completely at random across the available space and must systematically shrink its bounds until it converges exactly on this target peak.

# Code Structure
For educational clarity, the script is linearly organized into 6 distinct sections:

1. Settings: Defines the problem's dimensions, live points, and MCMC iterations.
2. Likelihood & Prior: Establishes the strict mathematical boundaries and the target Gaussian distribution.
3. Initial Live Points: The initial random scattering and scoring of the active points.
4. Slice Sampling Engine: The mathematical function responsible for safely finding the next valid point.
5. Main Nested Sampling Loop: The core loop that kills the worst point, finds a replacement via a random walk, and adapts the covariance matrix.
6. Visualization & Postprocessing: Visualizes the sampler's journey and extracts the final mathematical evidence.

# Dependencies
numpy
scipy
matplotlib
tqdm
anesthetic
