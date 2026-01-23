import torch
from rlcd.scoring import Scorer


def test_mle_w_implementations_match():
    """
    Test that est_mle_W, est_mle_W_lbfgs and est_mle_W_irls produce similar results
    within a small float tolerance.
    """
    # Create synthetic data
    torch.manual_seed(42)
    n, d = 100, 5
    X = torch.randn(n, d)
    
    # Create a DAG mask (sparse)
    s = torch.zeros((d, d))
    s[0, 1] = 1
    s[0, 2] = 1
    s[1, 2] = 1
    s[1, 3] = 1
    s[2, 4] = 1
    
    # Create scorer
    scorer = Scorer(X)
    
    # Estimate W using all three methods
    W_adam = scorer.est_mle_W_adam(s, lr=0.05, tol=1e-6, max_steps=10000, verbose=True)
    W_lbfgs = scorer.est_mle_W_lbfgs(s, verbose=True)
    W_irls = scorer.est_mle_W_irls(s, verbose=True)
    
    # Compare results
    print(f"\nW_adam:\n{W_adam}")
    print(f"\nW_lbfgs:\n{W_lbfgs}")
    print(f"\nW_irls:\n{W_irls}")
    print(f"\nDifference (adam vs irls):\n{(W_adam - W_irls).abs()}")
    print(f"\nMax absolute difference (adam vs irls): {(W_adam - W_irls).abs().max().item():.2e}")
    print(f"\nDifference (lbfgs vs irls):\n{(W_lbfgs - W_irls).abs()}")
    print(f"\nMax absolute difference (lbfgs vs irls): {(W_lbfgs - W_irls).abs().max().item():.2e}")
    
    # Assert they are close within tolerance
    torch.testing.assert_close(W_adam, W_irls, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(W_lbfgs, W_irls, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(W_adam, W_lbfgs, rtol=1e-4, atol=1e-4)




if __name__ == "__main__":
    test_mle_w_implementations_match()
