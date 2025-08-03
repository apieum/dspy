"""Test ScoreMatrix.update_scores() method."""

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort
from dspy.teleprompt.gepa.data.score_matrix import ScoreMatrix


def test_update_scores_adds_new_candidates():
    """Test that update_scores adds new candidates to the matrix."""
    score_matrix = ScoreMatrix()
    
    # Create test candidates with scores
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=0)
    candidate1.task_scores = [0.8, 0.6, 0.0]  # Good on task 0, ok on task 1, bad on task 2
    
    module2 = dspy.Predict("input -> output") 
    candidate2 = Candidate(module2, generation_number=0)
    candidate2.task_scores = [0.5, 0.9, 0.7]  # Ok on task 0, great on task 1, good on task 2
    
    # Update matrix with cohort
    cohort = Cohort(candidate1, candidate2)
    score_matrix.update_scores(cohort)
    
    # Check that best candidates are correctly assigned
    assert score_matrix.get_best_candidate_for_task(0) == candidate1  # 0.8 > 0.5
    assert score_matrix.get_best_candidate_for_task(1) == candidate2  # 0.9 > 0.6
    assert score_matrix.get_best_candidate_for_task(2) == candidate2  # 0.7 > 0.0
    
    assert set(score_matrix.get_all_task_ids()) == {0, 1, 2}


def test_update_scores_replaces_worse_candidates():
    """Test that update_scores replaces existing candidates with better ones."""
    score_matrix = ScoreMatrix()
    
    # Initial candidate
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=0)
    candidate1.task_scores = [0.5, 0.5]
    cohort1 = Cohort(candidate1)
    score_matrix.update_scores(cohort1)
    
    # Better candidate
    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=1)
    candidate2.task_scores = [0.8, 0.3]  # Better on task 0, worse on task 1
    
    cohort2 = Cohort(candidate2)
    score_matrix.update_scores(cohort2)
    
    # Check replacements
    assert score_matrix.get_best_candidate_for_task(0) == candidate2  # Better score
    assert score_matrix.get_best_candidate_for_task(1) == candidate1  # Kept original


if __name__ == "__main__":
    test_update_scores_adds_new_candidates()
    test_update_scores_replaces_worse_candidates()
    print("ScoreMatrix.update_scores() tests passed!")