"""Test CandidatePool.promote() method."""

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort
from dspy.teleprompt.gepa.data.candidate_pool import CandidatePool


def test_promote_adds_cohort_to_pool():
    """Test that promote() adds all candidates from cohort to pool."""
    pool = CandidatePool()
    
    # Create candidates with scores
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=1)
    candidate1.task_scores = [0.8, 0.6]
    
    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=1)
    candidate2.task_scores = [0.5, 0.9]
    
    cohort = Cohort(candidate1, candidate2)
    
    # Promote cohort to pool
    pool.promote(cohort)
    
    # Check that candidates were added
    assert len(pool.candidates) == 2
    assert candidate1 in pool.candidates
    assert candidate2 in pool.candidates
    
    # Check generation indexing
    assert len(pool.candidates_by_generation[1]) == 2
    assert candidate1 in pool.candidates_by_generation[1]
    assert candidate2 in pool.candidates_by_generation[1]


def test_promote_updates_score_matrix():
    """Test that promote() updates the score matrix correctly."""
    pool = CandidatePool()
    
    # Create candidates with scores
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=0)
    candidate1.task_scores = [0.8, 0.6]
    
    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=0)
    candidate2.task_scores = [0.5, 0.9]
    
    cohort = Cohort(candidate1, candidate2)
    
    # Promote cohort to pool
    pool.promote(cohort)
    
    # Check that score matrix was updated
    assert pool.score_matrix.task_scores[0] == candidate1  # 0.8 > 0.5
    assert pool.score_matrix.task_scores[1] == candidate2  # 0.9 > 0.6
    assert set(pool.score_matrix.task_scores.keys()) == {0, 1}


def test_promote_multiple_cohorts():
    """Test promoting multiple cohorts to pool."""
    pool = CandidatePool()
    
    # First cohort
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=0)
    candidate1.task_scores = [0.5, 0.5]
    cohort1 = Cohort(candidate1)
    pool.promote(cohort1)
    
    # Second cohort with better candidate
    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=1)
    candidate2.task_scores = [0.8, 0.3]
    cohort2 = Cohort(candidate2)
    pool.promote(cohort2)
    
    # Check total candidates
    assert len(pool.candidates) == 2
    assert len(pool.candidates_by_generation[0]) == 1
    assert len(pool.candidates_by_generation[1]) == 1
    
    # Check score matrix updates
    assert pool.score_matrix.task_scores[0] == candidate2  # Better score
    assert pool.score_matrix.task_scores[1] == candidate1  # Kept original


if __name__ == "__main__":
    test_promote_adds_cohort_to_pool()
    test_promote_updates_score_matrix() 
    test_promote_multiple_cohorts()
    print("CandidatePool.promote() tests passed!")