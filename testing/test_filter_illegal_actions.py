import torch
import pytest
from rlcd.actions import filter_illegal_actions_bruteforce, filter_illegal_actions


class TestFilterIllegalActions:
    """Test that both implementations produce identical results."""

    def test_empty_graph(self):
        """Test with an empty DAG (no edges)."""
        s = torch.zeros((3, 3))
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        print((~torch.eye(3, dtype=torch.bool)).int())
        assert torch.equal(result_bruteforce, ~torch.eye(3, dtype=torch.bool))
        assert torch.equal(result_bruteforce, result_optimized)

    def test_single_node(self):
        """Test with a single node."""
        s = torch.zeros((1, 1))
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_two_nodes_no_edges(self):
        """Test with two nodes and no edges."""
        s = torch.zeros((2, 2))
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_two_nodes_with_edge(self):
        """Test with two nodes, one edge."""
        s = torch.zeros((2, 2))
        s[0, 1] = 1  # edge from 0 to 1
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_three_node_chain(self):
        """Test with a chain: 0 -> 1 -> 2."""
        s = torch.zeros((3, 3))
        s[0, 1] = 1
        s[1, 2] = 1
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_complex_dag(self):
        """Test with a more complex DAG."""
        s = torch.zeros((4, 4))
        s[0, 1] = 1
        s[0, 2] = 1
        s[1, 3] = 1
        s[2, 3] = 1
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_diamond_dag(self):
        """Test with a diamond-shaped DAG: 0 -> {1,2} -> 3."""
        s = torch.zeros((4, 4))
        s[0, 1] = 1
        s[0, 2] = 1
        s[1, 3] = 1
        s[2, 3] = 1
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_larger_graph(self):
        """Test with a larger random DAG."""
        torch.manual_seed(42)
        d = 5
        s = torch.zeros((d, d))
        # Create a random DAG by only adding edges in the upper triangle
        for i in range(d):
            for j in range(i + 1, d):
                if torch.rand(1).item() > 0.5:
                    s[i, j] = 1
        
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        assert torch.equal(result_bruteforce, result_optimized)

    def test_self_loop_forbidden(self):
        """Test that self-loops are forbidden in both implementations."""
        s = torch.zeros((3, 3))
        s[0, 1] = 1
        
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        
        # Diagonal should all be False (self-loops are illegal)
        assert not result_bruteforce[0, 0]
        assert not result_bruteforce[1, 1]
        assert not result_bruteforce[2, 2]
        assert not result_optimized[0, 0]
        assert not result_optimized[1, 1]
        assert not result_optimized[2, 2]

    def test_length_2_loops_forbidden(self):
        """Test that length-2 cycles are forbidden."""
        s = torch.zeros((3, 3))
        s[0, 1] = 1
        
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        
        # Edge (1, 0) would create a cycle with (0, 1), so should be False
        assert not result_bruteforce[1, 0]
        assert not result_optimized[1, 0]

    def test_transitive_cycle_forbidden(self):
        """Test that adding edges creating transitive cycles are forbidden."""
        s = torch.zeros((3, 3))
        s[0, 1] = 1
        s[1, 2] = 1
        
        result_bruteforce = filter_illegal_actions_bruteforce(s)
        result_optimized = filter_illegal_actions(s)
        
        # Edge (2, 0) would create a cycle: 0 -> 1 -> 2 -> 0
        assert not result_bruteforce[2, 0]
        assert not result_optimized[2, 0]

    def test_consistency_across_sizes(self):
        """Test consistency for various graph sizes."""
        for size in range(1, 6):
            s = torch.zeros((size, size))
            # Add some random edges
            for i in range(size):
                for j in range(i + 1, size):
                    if torch.rand(1).item() > 0.6:
                        s[i, j] = 1
            
            result_bruteforce = filter_illegal_actions_bruteforce(s)
            result_optimized = filter_illegal_actions(s)
            assert torch.equal(result_bruteforce, result_optimized), \
                f"Mismatch for graph size {size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
