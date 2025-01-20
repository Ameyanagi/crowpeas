import os
import psutil

def get_optimal_num_workers():
    # Get number of CPU cores
    num_cpus = os.cpu_count()
    
    # Get system memory info
    mem = psutil.virtual_memory()
    available_memory_gb = mem.available / (1024**3)  # Convert to GB
    
    # Heuristic: Use either number of CPUs or scale based on available memory
    # Leaving some headroom for the OS and other processes
    suggested_workers = min(
        num_cpus - 2,  # Leave two CPU core free
        int(available_memory_gb / 2)  # Assume each worker might need ~2GB
    )
    
    # Ensure at least 1 worker and no more than 16 (arbitrary upper limit)
    return max(1, min(suggested_workers, 32))

if __name__ == "__main__":
    num_workers = get_optimal_num_workers()
    print(f"Optimal number of workers: {num_workers}")
    print(f"System has {os.cpu_count()} CPUs and {psutil.virtual_memory().available / (1024**3):.2f} GB available memory")