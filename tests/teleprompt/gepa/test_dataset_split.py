"""Test GEPA Algorithm 1 dataset split implementation."""

import pytest
from unittest.mock import Mock, patch

import dspy
from dspy.teleprompt.gepa.core import GEPA
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider


class TestDatasetSplit:
    """Test GEPA Algorithm 1 compliant dataset splitting."""

    def test_compile_splits_dataset_correctly(self):
        """Test that compile() splits dataset into D_feedback and D_pareto as per Algorithm 1."""
        
        # Create mock components to capture what datasets they receive
        mock_budget = Mock()
        mock_selector = Mock()
        mock_generator = Mock()
        mock_evaluator = Mock()
        
        gepa = GEPA(
            budget=mock_budget,
            selector=mock_selector, 
            generator=mock_generator,
            evaluator=mock_evaluator
        )
        
        # Create test dataset
        training_data = [dspy.Example(input=f'example_{i}', output=f'answer_{i}') for i in range(10)]
        
        # Create simple student
        student = Mock(spec=dspy.Module)
        student.deepcopy = Mock(return_value=student)
        
        # Mock the optimization loop to avoid complex setup
        with patch.object(gepa, 'next_generation') as mock_next_gen:
            mock_next_gen.return_value = student
            
            # Mock evaluator and selector methods
            mock_evaluator.evaluate_for_promotion = Mock(return_value=Mock())
            mock_selector.promote = Mock(return_value=Mock())
            mock_selector.best_candidate = Mock(return_value=Mock(average_task_score=Mock(return_value=0.8)))
            
            # Compile with default split ratio (0.25)
            gepa.compile(student, training_data)
            
            # Verify start_compilation was called with split datasets
            mock_budget.start_compilation.assert_called_once()
            mock_selector.start_compilation.assert_called_once()
            mock_generator.start_compilation.assert_called_once()
            mock_evaluator.start_compilation.assert_called_once()
            
            # Check the arguments passed to start_compilation
            args = mock_generator.start_compilation.call_args[0]
            student_arg, d_feedback, d_pareto = args
            
            # Verify dataset sizes (25% for pareto, 75% for feedback)
            assert len(d_pareto) == 2  # 25% of 10 = 2.5 → 2
            assert len(d_feedback) == 8  # 75% of 10 = 7.5 → 8
            assert len(d_feedback) + len(d_pareto) == len(training_data)

    def test_compile_respects_custom_split_ratio(self):
        """Test that compile() respects custom pareto_split_ratio parameter."""
        
        # Create mock components
        mock_budget = Mock()
        mock_selector = Mock() 
        mock_generator = Mock()
        mock_evaluator = Mock()
        
        gepa = GEPA(
            budget=mock_budget,
            selector=mock_selector,
            generator=mock_generator, 
            evaluator=mock_evaluator
        )
        
        # Create test dataset
        training_data = [dspy.Example(input=f'example_{i}', output=f'answer_{i}') for i in range(20)]
        
        # Create simple student
        student = Mock(spec=dspy.Module)
        student.deepcopy = Mock(return_value=student)
        
        # Mock the optimization loop
        with patch.object(gepa, 'next_generation') as mock_next_gen:
            mock_next_gen.return_value = student
            
            # Mock evaluator and selector methods
            mock_evaluator.evaluate_for_promotion = Mock(return_value=Mock())
            mock_selector.promote = Mock(return_value=Mock())
            mock_selector.best_candidate = Mock(return_value=Mock(average_task_score=Mock(return_value=0.8)))
            
            # Compile with 40% split ratio
            gepa.compile(student, training_data, pareto_split_ratio=0.4)
            
            # Check the arguments passed to start_compilation
            args = mock_generator.start_compilation.call_args[0]
            student_arg, d_feedback, d_pareto = args
            
            # Verify dataset sizes (40% for pareto, 60% for feedback)
            assert len(d_pareto) == 8   # 40% of 20 = 8
            assert len(d_feedback) == 12  # 60% of 20 = 12
            assert len(d_feedback) + len(d_pareto) == len(training_data)

    def test_generator_receives_feedback_dataset(self):
        """Test that Generator receives D_feedback for minibatch sampling."""
        
        from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
        from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
        
        # Create feedback provider with simple metric
        feedback_provider = FeedbackProvider(metric=lambda ex, pred, trace=None: 0.5)
        
        # Create ReflectivePromptMutation generator
        generator = ReflectivePromptMutation(feedback_provider)
        
        # Create datasets
        d_feedback = [dspy.Example(input=f'feedback_{i}', output=f'answer_{i}') for i in range(8)]
        d_pareto = [dspy.Example(input=f'pareto_{i}', output=f'answer_{i}') for i in range(2)]
        
        # Call start_compilation
        student = Mock(spec=dspy.Module)
        generator.start_compilation(student, d_feedback, d_pareto)
        
        # Verify generator uses D_feedback (not D_pareto)
        assert generator.feedback_data == d_feedback
        assert generator.feedback_data != d_pareto
        assert len(generator.feedback_data) == 8

    def test_evaluator_receives_pareto_dataset(self):
        """Test that Evaluator receives D_pareto for candidate evaluation."""
        
        from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
        
        # Create evaluator
        evaluator = PromotionEvaluator(metric=lambda ex, pred, trace=None: 0.5)
        
        # Create datasets  
        d_feedback = [dspy.Example(input=f'feedback_{i}', output=f'answer_{i}') for i in range(8)]
        d_pareto = [dspy.Example(input=f'pareto_{i}', output=f'answer_{i}') for i in range(2)]
        
        # Call start_compilation
        student = Mock(spec=dspy.Module)
        evaluator.start_compilation(student, d_feedback, d_pareto)
        
        # Verify evaluator uses D_pareto (not D_feedback) 
        assert evaluator.evaluation_data == d_pareto
        assert evaluator.evaluation_data != d_feedback
        assert len(evaluator.evaluation_data) == 2

    def test_dataset_split_prevents_overlap(self):
        """Test that D_feedback and D_pareto don't overlap."""
        
        # Create mock components
        mock_budget = Mock()
        mock_selector = Mock()
        mock_generator = Mock()
        mock_evaluator = Mock()
        
        gepa = GEPA(
            budget=mock_budget,
            selector=mock_selector,
            generator=mock_generator,
            evaluator=mock_evaluator
        )
        
        # Create test dataset with unique identifiers
        training_data = [dspy.Example(input=f'unique_example_{i}', output=f'answer_{i}') for i in range(10)]
        
        # Create simple student
        student = Mock(spec=dspy.Module)
        student.deepcopy = Mock(return_value=student)
        
        # Mock the optimization loop
        with patch.object(gepa, 'next_generation') as mock_next_gen:
            mock_next_gen.return_value = student
            
            # Mock evaluator and selector methods
            mock_evaluator.evaluate_for_promotion = Mock(return_value=Mock())
            mock_selector.promote = Mock(return_value=Mock())
            mock_selector.best_candidate = Mock(return_value=Mock(average_task_score=Mock(return_value=0.8)))
            
            # Compile
            gepa.compile(student, training_data)
            
            # Get the split datasets
            args = mock_generator.start_compilation.call_args[0]
            student_arg, d_feedback, d_pareto = args
            
            # Verify no overlap between datasets
            feedback_inputs = set(ex.input for ex in d_feedback)
            pareto_inputs = set(ex.input for ex in d_pareto)
            
            assert len(feedback_inputs.intersection(pareto_inputs)) == 0, "D_feedback and D_pareto should not overlap"
            assert len(feedback_inputs.union(pareto_inputs)) == len(training_data), "Combined datasets should equal original"

    def test_minimum_dataset_sizes(self):
        """Test that split ensures minimum of 1 example for D_pareto."""
        
        # Create mock components
        mock_budget = Mock()
        mock_selector = Mock()
        mock_generator = Mock()  
        mock_evaluator = Mock()
        
        gepa = GEPA(
            budget=mock_budget,
            selector=mock_selector,
            generator=mock_generator,
            evaluator=mock_evaluator
        )
        
        # Create very small dataset
        training_data = [dspy.Example(input='single_example', output='answer')]
        
        # Create simple student
        student = Mock(spec=dspy.Module)
        student.deepcopy = Mock(return_value=student)
        
        # Mock the optimization loop
        with patch.object(gepa, 'next_generation') as mock_next_gen:
            mock_next_gen.return_value = student
            
            # Mock evaluator and selector methods
            mock_evaluator.evaluate_for_promotion = Mock(return_value=Mock())
            mock_selector.promote = Mock(return_value=Mock())
            mock_selector.best_candidate = Mock(return_value=Mock(average_task_score=Mock(return_value=0.8)))
            
            # Compile
            gepa.compile(student, training_data)
            
            # Get the split datasets
            args = mock_generator.start_compilation.call_args[0]
            student_arg, d_feedback, d_pareto = args
            
            # Verify minimum sizes
            assert len(d_pareto) >= 1, "D_pareto should have at least 1 example"
            assert len(d_feedback) >= 0, "D_feedback can be empty but not negative"