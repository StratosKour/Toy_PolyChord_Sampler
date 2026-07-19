import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from tqdm import trange
from anesthetic import NestedSamples

# Here we are difining the seed of the random number function.
np.random.seed(0)

# Section 1 - Nested Sampling Settings.
"Our problems have parameters. Here we call them dimentions. These are very imporant for "
"what we are doing."

dimensions = 2  # Number of dimensions in the problem.
# This is not a "Rule of thumb" but is a good idea for picking the live points. The more dimensions you have the more live points you need.
live_points = 25 * dimensions
# Same logic as above. The more dimensions you have the more iterations you need.
iterations = dimensions * live_points

"Our sampler works by taking the number with the least likelihood, then picking a new points that has"
"a likelihood greater than the least likelihood. Using this point, the algorithm will take a random walk"
"mcmc_steps, which is to find a new points that has likelihood greater than the previous one."

# This is the number of steps the algorithm will take to find a new point.
mcmc_steps = 3 * dimensions


# Sections 2 - Likelihood and prior.
"This is our example. We know where the answer is, but we will pretend we don't."

mean = np.ones(dimensions) * 0.5  # This is the center of our distribution.

"To define a multivariate normal distribution(Gaussian distribution), the covariance matrix must STRICTLY"
"be symmetric, and positive semi definite. This is a requirement for the distribution to be valid."

# This is the covariance matrix of our distribution
covariance_matrix = np.random.rand(dimensions, dimensions)
# This is the corrected covariance matri
corrected_covarriance_matrix = 0.01 * covariance_matrix @ covariance_matrix.T

# This is the distribution we will be sampling from.
distribution = multivariate_normal(mean=mean, cov=corrected_covarriance_matrix)

"The above works like this:"
"def manual_multivariate_normal(x, mean, cov):"
"1. dimensions = len(mean) # Number of dimensions"
"2. determinant_covariance = np.linalg.det(covariance) # Calculating the determinant of the covariance matrix"
"3. Inverse_covariance = np.linalg.inv(covariance) # Calculating the inverse of the covariance matrix"
"4. difference = x - mean # Calculating the difference between the point and the mean"
"5. exponent = -0.5 * difference.T @ Inverse_covariance @ difference # Calculating the exponent"
"6. normalization = 1.0 / np.sqrt(((2 * np.pi) ** k) * det_cov) # Calculating the normalization factor"
"7. Probability_Denisity_Function = PDF = normalization * np.exp(exponent) # calculating the probabability density function"

"Now we need to evaluate how 'good' a point is. This is done by calculating the log likelihood of the point. "
"We use the logarithm of the PDF (logpdf) instead of the normal PDF because probabilities"
"in high dimensions get extremely close to zero. Computers struggle to calculate numbers like"
"0.00000000001, but taking the log turns these tiny probabilities into manageable negative numbers."


def log_likelihood(x):
    # This is the log likelihood of the point x. The higher the log likelihood, the better the point is.
    return distribution.logpdf(x)


"This algorithm might try to wander into areas of the parameter space that are not valid. To prevent this, we need to define a prior."
"1. (x > 0) & (x < 1) checks if the point is within the bounds of the prior. If it is, we return 0, which means the point is valid."
"2. .all(axis=-1) checks if all dimensions of the point are within the bounds of the prior. If they are, we return 0, which means the point is valid"
"3. np.where(((x > 0) & (x < 1)).all(axis=-1), 0, -np.inf) means:"
"- If the condition is met (inside the box), add 0 to the score (keep the score as is)."
"- If the condition is broken (outside the box), make the score Negative Infinity (-np.inf)."
"Negative Infinity in log-space means 0% probability. This physically forces the algorithm to stay inside the box."


def log_posterior_unnormalized(x):
    # Uniform prior between 0 and 1 for each dimension.
    # Here we get the distribution inside the box.
    return log_likelihood(x) + np.where(((x > 0) & (x < 1)).all(axis=-1), 0, -np.inf)


# Section 3 - Initial live points.
"Here we are setting up the starting board for our Nested Sampling algorithm."
"First, we scatter our initial 'live points' entirely at random withing our search space."

x = np.random.rand(live_points, dimensions)  # This is the initial live points.

scores_list = []

"Now we need to evaluate how 'good' each starting point is."
"We loop through every single point and pass it through our log_posterior_unnormalized function"
"This calculates its score based on both the likelihood and the prior."

for current_point in x:  # Loop to find the posterior of all the live points.
    current_score = log_posterior_unnormalized(current_point)
    # This is the initial scores of the live points.
    scores_list.append(current_score)

"Finally we collect all these calculated scores in a array."

# This is the initial scores of the live points in an array format.
array_of_scores = np.array(scores_list)

"Points must remember the score they had to beat to be 'born'. These first points"
"beat nothing, so their starting boundary is simply Negative Infinity (-np.inf)."

birth_scores = -np.inf * np.ones_like(array_of_scores)

"The algorithm needs a compass for its future 'random walks'. np.eye() creates an"
"Identity Matrix, setting it to explore each dimension equally and independently."

proposal_covariance = np.eye(dimensions)

"We create empty arrays as a 'graveyard'. As the algorithm kills the worst points,"
"we store their data here. These dead points will eventually build our final answer."

dead_points = np.empty((0, dimensions))
dead_scores = np.empty(0)
dead_birth_scores = np.empty(0)


# Section 4 - Slice sampling (Neal's stepping-out + shrinkage algorithm)
"The Slice Sampling function is the engine that find new, better points for our algorithm."
"When we kill the worst point, we need a replacement that has a STRICLTY HIGHER score"
"Insted of guessing blindly, this algorithm takes a 'slice' of that likelihood, of our target score."
"It picks a random direction and extends a line segment along that direcion until both ends drop bolow the"
"the target score. Then, it randomly picks a point on that line. If the point is valid, we keep it."
"If it missed, it shrinks the line towards the center and tries again."
"This guarantees we always find a better point efficiently."


def slice_sampling(starting_point, step_direction, target_score):
    "Create an initial search bracket around our starting point. Instead of placing the"
    "point perfectly in the center, we randomly shift the bracket. This asymmetry prevents"
    "the algorithm from getting stuck in repetitive, symmetric search patterns."

    # Randomly shift our initial search braket along the chosen direction
    random_shift = np.random.rand()
    positive_bound = starting_point + random_shift * step_direction
    negative_bound = starting_point + (random_shift - 1) * step_direction

    "step_direction acts as both a compass and a ruler. In a multi-dimensional space, it defines"
    "a specific, random 1D line and sets the base scale for our steps. The algorithm restricts"
    "its search (the expanding and shrinking) strictly along this single line."

    # Step out: Expand the boundaries outwards until the score drops below our target
    while log_posterior_unnormalized(positive_bound) > target_score:
        positive_bound += step_direction
    while log_posterior_unnormalized(negative_bound) > target_score:
        negative_bound -= step_direction

    "Pick a random point inside the bracket. If it is valid, we found our answer."
    "If it fails, shrink the boundary towards the starting point and try again."
    while True:
        shrink_factor = np.random.rand()
        proposed_point = negative_bound + shrink_factor * \
            (positive_bound - negative_bound)

        proposed_score = log_posterior_unnormalized(proposed_point)

        if proposed_score > target_score:
            return proposed_point, proposed_score

        # If the proposal fails, check which side of the start it laned on, and shrink that side
        if (proposed_point - starting_point) @ step_direction > 0:
            positive_bound = proposed_point
        else:
            negative_bound = proposed_point


# Section 5 - Main nested sampling loop
for _ in range(iterations):

    "5.a) Kill the worst point."
    worst_score = array_of_scores.min()

    "A 'mask' in Numpy is simply a True/False stencil used to filter data instantly."
    "First, we apply the dead_mask to pull out the single worst point and append its data"
    "to our permanent graveyard arrays. Then, we apply the survivors_mask to our active"
    "arrays. This overwrites them, effectively wiping the dead point from the playing board"
    "and moving forward with only the remaining winners."

    # Create a filter (True/False) to seperate the survivors from the dead.
    survivors_mask = array_of_scores > worst_score  # True or False operation
    dead_mask = ~survivors_mask  # The exact opposite of the survivors mas

    # Take the points that failed (dead mask) and add them to our graveyard arrays.
    dead_points = np.concatenate((dead_points, x[dead_mask]))
    dead_scores = np.concatenate((dead_scores, array_of_scores[dead_mask]))
    dead_birth_scores = np.concatenate(
        (dead_birth_scores, birth_scores[dead_mask]))

    # Keep ONLY the survivors to continue the process
    x = x[survivors_mask]
    array_of_scores = array_of_scores[survivors_mask]
    birth_scores = birth_scores[survivors_mask]

    "5.b) Build an adaptive proposal from current live + phantom points"

    # Prepare the mathematical compass for our random walk using the covariance matrix
    step_generator = multivariate_normal(
        np.zeros(dimensions), proposal_covariance)
    inverse_covariance = np.linalg.inv(proposal_covariance)

    # Keep a termporary copy of our surviving points. We will this to 'learn' the shape.
    # of the likelihood later and update our compass.
    phantom_points = x.copy()

    # Pick one random surviving point to act asthe starting point for our new search
    starting_point = x[np.random.choice(len(x))]

    "5.c) Chain of slice - sampling steps to decorrelate from starting_point"
    # Take a series of steps to find a new point. We take multiple steps so the final
    # point becomes completely independent from where it started.

    for _ in range(mcmc_steps):

        # Generate a radnom direction, then adjust its scale using our compass
        raw_direction = step_generator.rvs()
        step_direction = raw_direction / \
            (raw_direction @ inverse_covariance @ raw_direction) ** 0.5

        # Use the slice_sample function to find a new point alonf this direction
        # that has a score STRICTLY HIGHER than the worst_score
        new_point, new_score = slice_sampling(
            starting_point, step_direction, worst_score)

        # Add this intermediate point to our phantom list to help map the terrain.
        phantom_points = np.concatenate((phantom_points, [new_point]))

        starting_point = new_point

    " 5.d) Adapt the covariance, add the final point as new live point"

    "Update our compass (covariance matrix) based on the shape of all the phantom points"
    "we just explored. The algorithm gets smarter about the peaks shape over time."
    proposal_covariance = np.cov(phantom_points.T)

    "Add our completely new, successfully found point into the official live points list,"
    "along with its score and the boundary it had to beat (worst_score)."
    x = np.concatenate((x, [new_point]))
    array_of_scores = np.concatenate((array_of_scores, [new_score]))
    birth_scores = np.concatenate((birth_scores, [worst_score]))


# Section 6 — Visualization and Postprocessing
"Before using anesthetic, it is incredibly helpful to visualize WHAT the algorithm"
"actually did in our 2D space, and HOW it climbed the probability mountain."

# Plot 1: The Shrinking Process (2D Scatter)
"Since our problem is strictly 2D, we can plot the exact coordinates of every point."
"Dead points (blue) show the footprint of the algorithm as it explored and shrank."
"Final live points (red) show the exact peak it finally squeezed into."

plt.figure(figsize=(8, 8))
plt.scatter(dead_points[:, 0], dead_points[:, 1], color='royalblue',
            alpha=0.3, s=10, label='Dead Points (Exploration Path)')
plt.scatter(x[:, 0], x[:, 1], color='red', alpha=0.8, s=40,
            label='Final Surviving Points (The Peak)')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Nested Sampling: Shrinking towards the True Mean (0.5, 0.5)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot 2: Climbing the Hill (Trace Plot)
"This plot proves the core rule of Nested Sampling: The worst score MUST strictly increase."
"You will see the line start low and curve upwards as the box gets tighter around the peak."

plt.figure(figsize=(10, 5))
plt.plot(dead_scores, color='purple', linewidth=2)
plt.title("Evolution of the Worst Score (Log-Likelihood) over Time")
plt.xlabel("Iteration")
plt.ylabel("Dead Point Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot 3 & 4: Postprocessing with Anesthetic (Professor's Code)
"To get the final mathematical answer (the Evidence), we must merge the points that died"
"along the way with the final survivors that hold the remaining probability volume."

final_all_points = np.concatenate((dead_points, x))
final_all_scores = np.concatenate((dead_scores, array_of_scores))
final_all_births = np.concatenate((dead_birth_scores, birth_scores))

samples = NestedSamples(
    final_all_points,
    logL=final_all_scores,
    logL_birth=final_all_births
)

# Plot 5: The Bell Curves (1D Posteriors)
"This visualizes the final probability curves for each dimension."
"A red dashed line exactly at 0.5 acts as our crosshair to verify the peak."

axes = samples.plot_1d([0, 1], figsize=(12, 5))
axes[0].axvline(0.5, color='red', linestyle='--',
                linewidth=2, label='True Center (0.5)')
axes[0].set_title("Dimension 1 (X Axis)")
axes[0].legend()
axes[1].axvline(0.5, color='red', linestyle='--',
                linewidth=2, label='True Center (0.5)')
axes[1].set_title("Dimension 2 (Y Axis)")
axes[1].legend()

plt.suptitle("Visual Verification: Do the distributions peak at 0.5?")
plt.tight_layout()
plt.show()

"1. Histogram of the Log-Evidence (logZ)"
"This shows our algorithm's final answer: the calculated total volume of the mountain."
samples.logZ(1000).hist(bins=50, alpha=0.7, color='teal')
plt.axvline(samples.logZ(), color='k', linestyle='dashed', label='Mean logZ')
plt.title("Distribution of Log-Evidence (logZ)")
plt.xlabel("logZ")
plt.ylabel("Frequency")
plt.legend()
plt.show()

"2. Interactive GUI"
"This opens the interactive window to explore the 1D and 2D posteriors."
samples.gui()
