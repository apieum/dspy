"""Test configuration-driven ReflectivePromptMutation design."""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Tuple

import dspy
from dspy.teleprompt.gepa.generation.config import ReflectiveMutationConfig, ModuleSelectionStrategy
from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.generation.reflection_strategy import GEPAReflection
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns


class TestReflectiveMutationConfig:
    """Test the configuration object itself."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = ReflectiveMutationConfig()
        
        assert config.minibatch_size == 5
        assert config.module_selection_strategy == ModuleSelectionStrategy.WORST_PERFORMING
        assert config.requires_improvement == True
        assert config.max_retries == 3
        assert config.minimum_score_threshold == 0.0
        assert config.improvement_threshold == 0.01
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        ReflectiveMutationConfig(
            minibatch_size=3,
            max_retries=2,
            minimum_score_threshold=0.5,
            improvement_threshold=0.1
        )
        
        # Invalid configs should raise
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig(minibatch_size=0)
        
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReflectiveMutationConfig(max_retries=-1)
        
        with pytest.raises(ValueError, match="minimum_score_threshold must be between 0.0 and 1.0"):
            ReflectiveMutationConfig(minimum_score_threshold=1.5)
        
        with pytest.raises(ValueError, match="improvement_threshold must be non-negative"):
            ReflectiveMutationConfig(improvement_threshold=-0.1)
        
        with pytest.raises(ValueError, match="selection_temperature must be positive"):
            ReflectiveMutationConfig(selection_temperature=0.0)
    
    def test_config_factory_methods(self):
        """Test configuration factory methods."""
        # Quick experiments config
        quick_config = ReflectiveMutationConfig.for_quick_experiments()
        assert quick_config.minibatch_size == 3
        assert quick_config.max_retries == 1
        assert quick_config.requires_improvement == False
        assert quick_config.module_selection_strategy == ModuleSelectionStrategy.RANDOM
        
        # Production config
        prod_config = ReflectiveMutationConfig.for_production()
        assert prod_config.minibatch_size == 8
        assert prod_config.max_retries == 5
        assert prod_config.requires_improvement == True
        assert prod_config.improvement_threshold == 0.05
        assert prod_config.minimum_score_threshold == 0.1
        
        # Debugging config
        debug_config = ReflectiveMutationConfig.for_debugging()
        assert debug_config.enable_detailed_logging == True
        assert debug_config.minibatch_size == 2
        assert debug_config.preserve_original_on_failure == True
    
    def test_config_factory_overrides(self):
        """Test that factory methods accept overrides."""
        config = ReflectiveMutationConfig.for_quick_experiments(
            minibatch_size=10,
            requires_improvement=True
        )
        
        assert config.minibatch_size == 10  # Overridden
        assert config.requires_improvement == True  # Overridden
        assert config.max_retries == 1  # Default from factory
        assert config.module_selection_strategy == ModuleSelectionStrategy.RANDOM  # Default from factory


class TestConfigurationDrivenReflectivePromptMutation:
    """Test the refactored ReflectivePromptMutation with configuration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feedback_provider = Mock(spec=FeedbackProvider)
        self.feedback_provider.evaluate.return_value = (0.8, "Good performance")
        
        # Create test data
        self.feedback_data = [
            dspy.Example(question="test1", answer="answer1"),
            dspy.Example(question="test2", answer="answer2"),
            dspy.Example(question="test3", answer="answer3")
        ]
        
        # Create test module
        self.test_module = Mock(spec=dspy.Module)
        self.test_module.predictors.return_value = [Mock(), Mock()]  # 2 predictors
        
    def test_config_initialization(self):
        """Test initializing ReflectivePromptMutation with config."""
        config = ReflectiveMutationConfig(
            minibatch_size=3,
            module_selection_strategy=ModuleSelectionStrategy.RANDOM,
            requires_improvement=False
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        assert generator.config == config
        assert generator.feedback_provider == self.feedback_provider
        assert generator.config.minibatch_size == 3
        assert generator.config.requires_improvement == False
    
    def test_backward_compatibility_initialization(self):
        """Test that legacy initialization parameters still work."""
        generator = ReflectivePromptMutation(
            feedback_provider=self.feedback_provider,
            minibatch_size=4,
            module_selection="round_robin"
        )
        
        assert generator.feedback_provider == self.feedback_provider
        assert generator.config.minibatch_size == 4
        assert generator.config.module_selection_strategy == ModuleSelectionStrategy.ROUND_ROBIN
    
    def test_config_precedence_over_legacy_params(self):
        """Test that config parameters take precedence over legacy ones."""
        config = ReflectiveMutationConfig(
            minibatch_size=10,
            feedback_provider=self.feedback_provider
        )
        
        different_provider = Mock()
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=different_provider,  # Should be overridden by config
            minibatch_size=5  # Should be overridden by config
        )
        
        assert generator.feedback_provider == self.feedback_provider  # From config
        assert generator.config.minibatch_size == 10  # From config
    
    def test_module_selection_strategies(self):
        """Test different module selection strategies."""
        config = ReflectiveMutationConfig(
            module_selection_strategy=ModuleSelectionStrategy.ROUND_ROBIN
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        # Test round-robin
        assert generator._select_target_module(3) == 0
        assert generator._select_target_module(3) == 1
        assert generator._select_target_module(3) == 2
        assert generator._select_target_module(3) == 0  # Wraps around
    
    def test_worst_performing_module_selection(self):
        """Test worst-performing module selection strategy."""
        config = ReflectiveMutationConfig(
            module_selection_strategy=ModuleSelectionStrategy.WORST_PERFORMING
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        # Initially no history, should use round-robin fallback
        first_selection = generator._select_target_module(3)
        assert 0 <= first_selection < 3
        
        # Add performance history
        generator.module_performance_history = {
            0: [0.9, 0.8, 0.85],  # Good performance
            1: [0.3, 0.4, 0.2],   # Bad performance
            2: [0.7, 0.6, 0.8]    # Medium performance
        }
        
        # Should select module 1 (worst performing)
        worst_module = generator._select_worst_performing_module(3)
        assert worst_module == 1
    
    def test_minibatch_size_configuration(self):
        """Test that minibatch size is respected from config."""
        config = ReflectiveMutationConfig(minibatch_size=2)
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        generator.feedback_data = self.feedback_data  # 3 examples
        
        minibatch = generator._sample_minibatch()
        assert len(minibatch) == 2  # Respects config minibatch_size
    
    def test_improvement_validation_with_config(self):
        """Test improvement validation respects config settings."""
        # Config that doesn't require improvement
        config = ReflectiveMutationConfig(
            requires_improvement=False,
            minimum_score_threshold=0.5
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        # Mock the evaluation to return scores above threshold
        with patch.object(generator, '_ensure_evolvable') as mock_evolvable:
            mock_parent = Mock()
            mock_parent.collect_traces_and_evaluate.return_value = Mock(scores=[0.6])
            
            mock_child = Mock()
            mock_child.collect_traces_and_evaluate.return_value = Mock(scores=[0.7])
            
            mock_evolvable.return_value = mock_parent
            
            parent_candidate = Mock()
            parent_candidate.module = Mock()
            
            # Should accept because it meets minimum threshold and doesn't require improvement
            result = generator._validate_improvement(
                parent_candidate, mock_child, [], 0
            )
            
            assert result == True
    
    def test_improvement_threshold_validation(self):
        """Test that improvement threshold is properly enforced."""
        config = ReflectiveMutationConfig(
            requires_improvement=True,
            improvement_threshold=0.1,  # Require 0.1 improvement
            minimum_score_threshold=0.0
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        with patch.object(generator, '_ensure_evolvable') as mock_evolvable:
            mock_parent = Mock()
            mock_parent.collect_traces_and_evaluate.return_value = Mock(scores=[0.5])
            
            mock_child = Mock()
            # Child score 0.55 is only 0.05 improvement (below 0.1 threshold)
            mock_child.collect_traces_and_evaluate.return_value = Mock(scores=[0.55])
            
            mock_evolvable.return_value = mock_parent
            
            parent_candidate = Mock()
            parent_candidate.module = Mock()
            
            result = generator._validate_improvement(
                parent_candidate, mock_child, [], 0
            )
            
            assert result == False  # Insufficient improvement
    
    def test_minimum_score_threshold_validation(self):
        """Test that minimum score threshold is enforced."""
        config = ReflectiveMutationConfig(
            minimum_score_threshold=0.7,
            requires_improvement=False
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        with patch.object(generator, '_ensure_evolvable') as mock_evolvable:
            mock_parent = Mock()
            mock_parent.collect_traces_and_evaluate.return_value = Mock(scores=[0.5])
            
            mock_child = Mock()
            mock_child.collect_traces_and_evaluate.return_value = Mock(scores=[0.6])  # Below 0.7 threshold
            
            mock_evolvable.return_value = mock_parent
            
            parent_candidate = Mock()
            parent_candidate.module = Mock()
            
            result = generator._validate_improvement(
                parent_candidate, mock_child, [], 0
            )
            
            assert result == False  # Below minimum threshold
    
    def test_performance_history_tracking(self):
        """Test that performance history is properly tracked."""
        config = ReflectiveMutationConfig()
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        with patch.object(generator, '_ensure_evolvable') as mock_evolvable:
            mock_parent = Mock()
            mock_parent.collect_traces_and_evaluate.return_value = Mock(scores=[0.5])
            
            mock_child = Mock()
            mock_child.collect_traces_and_evaluate.return_value = Mock(scores=[0.8])
            
            mock_evolvable.return_value = mock_parent
            
            parent_candidate = Mock()
            parent_candidate.module = Mock()
            
            # Call validation which should update performance history
            generator._validate_improvement(parent_candidate, mock_child, [], 0)
            
            # Check that performance history was updated
            assert 0 in generator.module_performance_history
            assert generator.module_performance_history[0] == [0.8]
    
    def test_detailed_logging_configuration(self):
        """Test that detailed logging can be enabled via config."""
        config = ReflectiveMutationConfig(enable_detailed_logging=True)
        
        with patch('dspy.teleprompt.gepa.generation.reflective_mutation_native.logger') as mock_logger:
            generator = ReflectivePromptMutation(
                config=config,
                feedback_provider=self.feedback_provider
            )
            
            # Should have set logger level to DEBUG
            mock_logger.setLevel.assert_called_with(logging.DEBUG)
    
    def test_config_integration_with_generation(self):
        """Test full integration of config with generation process."""
        config = ReflectiveMutationConfig.for_quick_experiments(
            minibatch_size=1,
            requires_improvement=False
        )
        
        generator = ReflectivePromptMutation(
            config=config,
            feedback_provider=self.feedback_provider
        )
        
        # Set up data
        generator.start_compilation(Mock(), self.feedback_data, [])
        
        # Create mock parent
        parent = Candidate(
            module=self.test_module,
            generation_number=0,
            parents=[],
            creation_metadata={}
        )
        parents = Parents([parent])
        
        # Mock the necessary methods for full generation
        with patch.object(generator, '_ensure_evolvable') as mock_ensure, \
             patch.object(generator, '_validate_improvement', return_value=True) as mock_validate:
            
            mock_evolvable = Mock()
            mock_evolvable.predictors.return_value = [Mock(), Mock()]
            mock_evolvable.collect_traces_and_evaluate.return_value = Mock(scores=[0.7])
            mock_evolvable.evolve.return_value = Mock()
            mock_ensure.return_value = mock_evolvable
            
            # Generate should work with config
            result = generator.generate(parents)
            
            # Should return NewBorns (not empty due to mocked validation)
            assert isinstance(result, NewBorns)


class TestConfigurationExperimentation:
    """Test how the configuration enables easier experimentation."""
    
    def test_quick_vs_production_configs(self):
        """Demonstrate difference between quick and production configs."""
        quick_config = ReflectiveMutationConfig.for_quick_experiments()
        prod_config = ReflectiveMutationConfig.for_production()
        
        # Quick config is optimized for speed
        assert quick_config.minibatch_size < prod_config.minibatch_size
        assert quick_config.max_retries < prod_config.max_retries
        assert quick_config.requires_improvement == False  # Accept any mutation
        
        # Production config is optimized for quality
        assert prod_config.requires_improvement == True
        assert prod_config.improvement_threshold > 0
        assert prod_config.minimum_score_threshold > 0
    
    def test_custom_experimental_config(self):
        """Test creating custom experimental configurations."""
        # Configuration for testing aggressive mutation
        aggressive_config = ReflectiveMutationConfig(
            minibatch_size=10,  # More feedback data
            module_selection_strategy=ModuleSelectionStrategy.WORST_PERFORMING,
            requires_improvement=True,
            improvement_threshold=0.2,  # Require significant improvement
            max_retries=10  # More attempts
        )
        
        # Configuration for conservative mutation
        conservative_config = ReflectiveMutationConfig(
            minibatch_size=3,  # Less feedback data
            module_selection_strategy=ModuleSelectionStrategy.ROUND_ROBIN,
            requires_improvement=True,
            improvement_threshold=0.01,  # Accept small improvements
            minimum_score_threshold=0.8,  # High quality bar
            max_retries=2  # Fewer attempts
        )
        
        # Both should be valid but have different behaviors
        assert aggressive_config.improvement_threshold > conservative_config.improvement_threshold
        assert aggressive_config.minibatch_size > conservative_config.minibatch_size
        assert conservative_config.minimum_score_threshold > aggressive_config.minimum_score_threshold