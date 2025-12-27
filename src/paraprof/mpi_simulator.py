"""
MPI Communication Simulator for Testing Scalability

Wraps MPI communicator to add synthetic network latency and bandwidth limits,
allowing developers to emulate large-scale HPC performance on laptops.
"""
import time
import numpy as np
from mpi4py import MPI


class MPISimulator:
    """
    Wrapper around MPI communicator that adds configurable network delays.

    Use this to simulate realistic HPC network performance (e.g., InfiniBand,
    Ethernet) when testing on a laptop with limited cores.

    Parameters
    ----------
    comm : MPI.Comm
        The actual MPI communicator
    latency_us : float
        One-way network latency in microseconds (default: 1.0 for InfiniBand)
    bandwidth_gbps : float
        Network bandwidth in Gbps (default: 100 for HDR InfiniBand)
    enable_delays : bool
        Enable/disable delays (useful for toggling via environment variable)
    jitter_factor : float
        Random jitter as fraction of latency (default: 0.1 = ±10%)

    Examples
    --------
    >>> # Simulate InfiniBand network (1us latency, 100 Gbps)
    >>> comm = MPISimulator(MPI.COMM_WORLD, latency_us=1.0, bandwidth_gbps=100)
    >>>
    >>> # Simulate 10GbE network (10us latency, 10 Gbps)
    >>> comm = MPISimulator(MPI.COMM_WORLD, latency_us=10.0, bandwidth_gbps=10)
    >>>
    >>> # Simulate high-latency cloud network
    >>> comm = MPISimulator(MPI.COMM_WORLD, latency_us=100.0, bandwidth_gbps=25)

    Notes
    -----
    Typical HPC network characteristics:
    - InfiniBand EDR/HDR: 0.5-2 μs latency, 100-200 Gbps
    - 10 GbE: 5-20 μs latency, 10 Gbps
    - Cloud (AWS, Azure): 50-200 μs latency, 10-100 Gbps
    """

    def __init__(self, comm, latency_us=1.0, bandwidth_gbps=100,
                 enable_delays=True, jitter_factor=0.1):
        self.comm = comm
        self.latency_sec = latency_us * 1e-6
        self.bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
        self.enable_delays = enable_delays
        self.jitter_factor = jitter_factor

        # Track statistics
        self.stats = {
            'send_count': 0,
            'recv_count': 0,
            'send_bytes': 0,
            'recv_bytes': 0,
            'total_delay_sec': 0.0
        }

    def _get_message_size(self, obj):
        """Estimate message size in bytes (rough approximation)."""
        try:
            import pickle
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback: assume moderate size
            return 1024

    def _add_delay(self, message_size_bytes, operation='send'):
        """Add synthetic network delay based on latency and bandwidth."""
        if not self.enable_delays:
            return

        # Add jitter to latency (simulate network variation)
        jitter = np.random.uniform(-self.jitter_factor, self.jitter_factor)
        latency = self.latency_sec * (1 + jitter)

        # Calculate transmission time based on message size and bandwidth
        transmission_time = message_size_bytes / self.bandwidth_bytes_per_sec

        # Total delay = latency + transmission time
        total_delay = latency + transmission_time

        # Actually sleep to simulate the delay
        time.sleep(total_delay)

        # Update statistics
        self.stats['total_delay_sec'] += total_delay
        self.stats[f'{operation}_count'] += 1
        self.stats[f'{operation}_bytes'] += message_size_bytes

    def send(self, obj, dest, tag=0):
        """Send with simulated network delay."""
        msg_size = self._get_message_size(obj)
        self._add_delay(msg_size, 'send')
        return self.comm.send(obj, dest=dest, tag=tag)

    def recv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        """Receive with simulated network delay."""
        result = self.comm.recv(source=source, tag=tag, status=status)
        msg_size = self._get_message_size(result)
        self._add_delay(msg_size, 'recv')
        return result

    def isend(self, obj, dest, tag=0):
        """Non-blocking send with simulated delay."""
        msg_size = self._get_message_size(obj)
        self._add_delay(msg_size, 'send')
        return self.comm.isend(obj, dest=dest, tag=tag)

    def irecv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        """Non-blocking receive (delay added when result is accessed)."""
        # Note: For true simulation, should delay on .wait() or .test()
        return self.comm.irecv(source=source, tag=tag)

    def Iprobe(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        """Probe with small delay (probing has minimal overhead)."""
        if self.enable_delays:
            time.sleep(self.latency_sec * 0.1)  # Small probe cost
        return self.comm.Iprobe(source=source, tag=tag, status=status)

    def bcast(self, obj, root=0):
        """Broadcast with simulated delay."""
        msg_size = self._get_message_size(obj)
        # Broadcast uses tree topology, so log(N) latency
        num_ranks = self.comm.Get_size()
        tree_depth = int(np.ceil(np.log2(num_ranks)))
        self._add_delay(msg_size * tree_depth / num_ranks, 'send')
        return self.comm.bcast(obj, root=root)

    def barrier(self):
        """Barrier with simulated collective delay."""
        if self.enable_delays:
            num_ranks = self.comm.Get_size()
            tree_depth = int(np.ceil(np.log2(num_ranks)))
            time.sleep(self.latency_sec * tree_depth * 2)  # Up and down tree
        return self.comm.barrier()

    # Delegate all other MPI communicator methods
    def __getattr__(self, name):
        return getattr(self.comm, name)

    def print_stats(self):
        """Print communication statistics (master only)."""
        if self.comm.Get_rank() == 0:
            print("\n" + "="*60)
            print("MPI Simulation Statistics")
            print("="*60)
            print(f"Network: {self.latency_sec*1e6:.1f} μs latency, "
                  f"{self.bandwidth_bytes_per_sec*8/1e9:.0f} Gbps bandwidth")
            print(f"Total simulated delay: {self.stats['total_delay_sec']:.3f} seconds")
            print(f"Send operations: {self.stats['send_count']:,} "
                  f"({self.stats['send_bytes']/1024/1024:.2f} MB)")
            print(f"Recv operations: {self.stats['recv_count']:,} "
                  f"({self.stats['recv_bytes']/1024/1024:.2f} MB)")
            if self.stats['send_count'] > 0:
                avg_msg_size = self.stats['send_bytes'] / self.stats['send_count']
                print(f"Average message size: {avg_msg_size:.0f} bytes")
            print("="*60 + "\n")


def get_simulated_communicator(network_type='infiniband', enable=True):
    """
    Factory function to create an MPI simulator with preset network profiles.

    Parameters
    ----------
    network_type : str
        One of: 'infiniband', '10gbe', 'gigabit', 'cloud', 'none'
    enable : bool
        Enable delays (can be controlled via env var: PARAPROF_SIMULATE_MPI)

    Returns
    -------
    MPISimulator
        Configured MPI communicator wrapper

    Examples
    --------
    >>> # In your run script:
    >>> from paraprof.mpi_simulator import get_simulated_communicator
    >>> comm = get_simulated_communicator('10gbe')  # Simulate 10GbE network
    """
    import os

    # Allow environment variable override
    enable = enable and (os.environ.get('PARAPROF_SIMULATE_MPI', '1') != '0')

    profiles = {
        'infiniband': {'latency_us': 1.0, 'bandwidth_gbps': 100},
        '10gbe': {'latency_us': 10.0, 'bandwidth_gbps': 10},
        'gigabit': {'latency_us': 50.0, 'bandwidth_gbps': 1},
        'cloud': {'latency_us': 100.0, 'bandwidth_gbps': 25},
        'none': {'latency_us': 0.0, 'bandwidth_gbps': 1000}
    }

    if network_type not in profiles:
        raise ValueError(f"Unknown network type '{network_type}'. "
                        f"Choose from: {list(profiles.keys())}")

    profile = profiles[network_type]
    enable_for_type = enable and (network_type != 'none')

    return MPISimulator(MPI.COMM_WORLD,
                       latency_us=profile['latency_us'],
                       bandwidth_gbps=profile['bandwidth_gbps'],
                       enable_delays=enable_for_type)
