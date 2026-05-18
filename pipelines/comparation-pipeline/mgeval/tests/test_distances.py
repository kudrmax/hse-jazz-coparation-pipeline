"""Тесты distances — порт mgeval/utils.py."""
import numpy as np

from mgeval.distances import c_dist, kl_dist, overlap_area


def test_c_dist_known():
    A = np.array([1.0, 2.0])
    B = np.array([[1.0, 2.0], [4.0, 6.0]])
    # distances: ||A-B[0]||=0, ||A-B[1]||=sqrt(9+16)=5
    np.testing.assert_allclose(c_dist(A, B), [0.0, 5.0])


def test_c_dist_single_target():
    A = np.array([0.0, 0.0])
    B = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(c_dist(A, B), [5.0])


def test_kl_same_distribution_near_zero():
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, 500)
    B = rng.normal(0, 1, 500)
    # KDE на 500-точечной выборке N(0,1) даёт KL ~0.05-0.15 за счёт finite-sample
    # шума. Threshold 0.2 — мягкий «KL близок к нулю», но не нулевой.
    assert kl_dist(A, B) < 0.2


def test_kl_different_shapes_diverges():
    """KL отличает разные ФОРМЫ распределения (важно: reference считает на
    разных сетках linspace_A vs linspace_B, поэтому реагирует на форму PDF,
    а не на абсолютный диапазон значений)."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, 500)  # gaussian
    B = rng.uniform(-3, 3, 500)  # uniform с тем же спаном
    # Reference-формула на different-grid'ах даёт KL ~0.25-0.3 для этой пары.
    # Threshold 0.2 — sanity «функция различает формы», не подгонка под точное число.
    assert kl_dist(A, B) > 0.2


def test_oa_same_distribution_near_one():
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, 500)
    B = rng.normal(0, 1, 500)
    assert overlap_area(A, B) > 0.8


def test_oa_disjoint_distributions_near_zero():
    """OA на distributions с непересекающимися диапазонами → ~0
    (квад-интегрирование min(pdf_A, pdf_B) на общем range [0, 101] —
    вне родных диапазонов pdf'ы уже спали к нулю)."""
    A = np.linspace(0, 1, 100)
    B = np.linspace(100, 101, 100)
    assert overlap_area(A, B) < 0.05
