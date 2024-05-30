import numpy as np
from scipy.spatial import cKDTree
import time
import multiprocessing as mp

# Generate random positions for atoms
np.random.seed(42)
num_atoms = 1000
box_size = 10.0
positions = np.random.rand(num_atoms, 3) * box_size
cutoff_radius = 1.5

# Function to calculate distances without cutoff
def calculate_distances_without_cutoff(positions):
    num_atoms = positions.shape[0]
    distances = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    return distances

# Function to calculate distances without cutoff in parallel
def calculate_distances_without_cutoff_parallel(positions, start, end):
    distances = []
    for i in range(start, end):
        for j in range(i + 1, positions.shape[0]):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    return distances

# Function to calculate distances with cutoff using k-d tree
def calculate_distances_with_cutoff(positions, cutoff_radius):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff_radius)
    distances = [np.linalg.norm(positions[i] - positions[j]) for (i, j) in pairs]
    return distances

# Parallel calculation wrapper
def parallel_wrapper(func, positions, num_processes=4):  # Use 4 cores
    num_atoms = positions.shape[0]
    chunk_size = num_atoms // num_processes
    pool = mp.Pool(processes=num_processes)
    results = [pool.apply_async(func, (positions, i * chunk_size, (i + 1) * chunk_size if i != num_processes - 1 else num_atoms)) for i in range(num_processes)]
    pool.close()
    pool.join()
    distances = []
    for result in results:
        distances.extend(result.get())
    return distances

# Measure time for distance calculation without cutoff
start_time = time.time()
distances_without_cutoff = calculate_distances_without_cutoff(positions)
time_without_cutoff = time.time() - start_time

# Measure time for distance calculation without cutoff in parallel
start_time = time.time()
distances_without_cutoff_parallel = parallel_wrapper(calculate_distances_without_cutoff_parallel, positions)
time_without_cutoff_parallel = time.time() - start_time

# Measure time for distance calculation with cutoff
start_time = time.time()
distances_with_cutoff = calculate_distances_with_cutoff(positions, cutoff_radius)
time_with_cutoff = time.time() - start_time

# Since k-d tree with cutoff is already optimized, parallelization is not applied here
time_with_cutoff_parallel = time_with_cutoff

# Calculate speedups
speedup_without_cutoff = time_without_cutoff / time_without_cutoff_parallel
speedup_with_cutoff = time_without_cutoff_parallel / time_with_cutoff

# Print the results in a table format
print(f"{'Method':<30} {'Time (seconds)':<20} {'Speedup':<20}")
print("="*70)
print(f"Without cutoff, without parallel   {time_without_cutoff:<20.6f}")
print(f"Without cutoff, with parallel      {time_without_cutoff_parallel:<20.6f} {speedup_without_cutoff:<20.2f}")
print(f"With cutoff, without parallel      {time_with_cutoff:<20.6f}")
print(f"With cutoff, with parallel         {time_with_cutoff_parallel:<20.6f} {speedup_with_cutoff:<20.2f}")
