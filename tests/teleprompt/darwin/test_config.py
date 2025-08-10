"""Test configuration classes for Darwin framework."""

import pytest
from unittest.mock import Mock

from dspy.teleprompt.darwin.generation.config import (
    ModuleSelectionStrategy,
    ReflectiveMutationConfig
)


class TestModuleSelectionStrategy:
    """Test the ModuleSelectionStrategy enum."""

    def test_enum_values(self):
        """Test that all expected strategy values exist."""
        assert ModuleSelectionStrategy.RANDOM.value == "random"
        assert ModuleSelectionStrategy.WORST_PERFORMING.value == "worst_performing"
        assert ModuleSelectionStrategy.ALL.value == "all"
        assert ModuleSelectionStrategy.ROUND_ROBIN.value == "round_robin"

    def test_enum_membership(self):
        """Test enum membership and iteration."""
        strategies = list(ModuleSelectionStrategy)
        assert len(strategies) == 4
        assert ModuleSelectionStrategy.RANDOM in strategies
        assert ModuleSelectionStrategy.WORST_PERFORMING in strategies
        assert ModuleSelectionStrategy.ALL in strategies
        assert ModuleSelectionStrategy.ROUND_ROBIN in strategies


class TestReflectiveMutationConfig:
    """Test ReflectiveMutationConfig dataclass."""

    def test_default_initialization(self):
        """Test config creation with default values."""
        config = ReflectiveMutationConfig()
        
        # Test core parameters
        assert config.minibatch_size == 5
        assert config.module_selection_strategy == ModuleSelectionStrategy.WORST_PERFORMING
        assert config.max_retries == 3
        
        # Test reflection configuration
        assert config.reflection_strategy is None
        assert config.feedback_provider is None
        assert config.enhanced_feedback_function is None
        
        # Test module selection parameters
        assert config.selection_temperature == 1.0
        assert config.max_modules_per_generation is None
        
        # Test debugging parameters
        assert config.enable_detailed_logging is False
        assert config.preserve_original_on_failure is True

    def test_custom_initialization(self):
        """Test config creation with custom values."""
        mock_strategy = Mock()
        mock_provider = Mock()
        mock_function = Mock()
        
        config = ReflectiveMutationConfig(
            minibatch_size=10,
            module_selection_strategy=ModuleSelectionStrategy.RANDOM,
            max_retries=2,
            reflection_strategy=mock_strategy,
            feedback_provider=mock_provider,
            enhanced_feedback_function=mock_function,
            selection_temperature=0.5,
            max_modules_per_generation=3,
            enable_detailed_logging=True,
            preserve_original_on_failure=False
        )
        
        assert config.minibatch_size == 10
        assert config.module_selection_strategy == ModuleSelectionStrategy.RANDOM
        assert config.max_retries == 2
        assert config.reflection_strategy == mock_strategy
        assert config.feedback_provider == mock_provider
        assert config.enhanced_feedback_function == mock_function
        assert config.selection_temperature == 0.5
        assert config.max_modules_per_generation == 3
        assert config.enable_detailed_logging is True
        assert config.preserve_original_on_failure is False

    def test_validation_positive_minibatch_size(self):
        """Test validation of minibatch_size parameter."""
        # Valid positive values should work
        config = ReflectiveMutationConfig(minibatch_size=1)
        assert config.minibatch_size == 1
        
        config = ReflectiveMutationConfig(minibatch_size=100)
        assert config.minibatch_size == 100
        
        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig(minibatch_size=0)
        
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig(minibatch_size=-1)
        
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig(minibatch_size=-10)

    def test_validation_non_negative_max_retries(self):
        """Test validation of max_retries parameter."""
        # Valid non-negative values should work
        config = ReflectiveMutationConfig(max_retries=0)
        assert config.max_retries == 0
        
        config = ReflectiveMutationConfig(max_retries=5)
        assert config.max_retries == 5
        
        # Invalid negative values should raise ValueError
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReflectiveMutationConfig(max_retries=-1)
        
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReflectiveMutationConfig(max_retries=-5)

    def test_validation_positive_selection_temperature(self):
        """Test validation of selection_temperature parameter."""
        # Valid positive values should work
        config = ReflectiveMutationConfig(selection_temperature=0.1)
        assert config.selection_temperature == 0.1
        
        config = ReflectiveMutationConfig(selection_temperature=2.0)
        assert config.selection_temperature == 2.0
        
        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="selection_temperature must be positive"):
            ReflectiveMutationConfig(selection_temperature=0.0)
        
        with pytest.raises(ValueError, match="selection_temperature must be positive"):
            ReflectiveMutationConfig(selection_temperature=-0.5)
        
        with pytest.raises(ValueError, match="selection_temperature must be positive"):
            ReflectiveMutationConfig(selection_temperature=-1.0)

    def test_validation_max_modules_per_generation(self):
        """Test validation of max_modules_per_generation parameter."""
        # None should be valid (no limit)
        config = ReflectiveMutationConfig(max_modules_per_generation=None)
        assert config.max_modules_per_generation is None
        
        # Positive values should work
        config = ReflectiveMutationConfig(max_modules_per_generation=1)
        assert config.max_modules_per_generation == 1
        
        config = ReflectiveMutationConfig(max_modules_per_generation=10)
        assert config.max_modules_per_generation == 10
        
        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="max_modules_per_generation must be positive or None"):
            ReflectiveMutationConfig(max_modules_per_generation=0)
        
        with pytest.raises(ValueError, match="max_modules_per_generation must be positive or None"):
            ReflectiveMutationConfig(max_modules_per_generation=-1)
        
        with pytest.raises(ValueError, match="max_modules_per_generation must be positive or None"):
            ReflectiveMutationConfig(max_modules_per_generation=-5)

    def test_validation_combined_invalid_parameters(self):
        """Test that validation catches multiple invalid parameters."""
        # Test that the first invalid parameter is caught
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig(
                minibatch_size=-1,
                max_retries=-1,
                selection_temperature=-1.0
            )

    def test_for_quick_experiments_factory(self):
        """Test the for_quick_experiments factory method."""
        config = ReflectiveMutationConfig.for_quick_experiments()
        
        assert config.minibatch_size == 3
        assert config.max_retries == 1
        assert config.module_selection_strategy == ModuleSelectionStrategy.RANDOM
        
        # Other defaults should remain unchanged
        assert config.selection_temperature == 1.0
        assert config.enable_detailed_logging is False

    def test_for_quick_experiments_with_overrides(self):
        """Test the for_quick_experiments factory with custom overrides."""
        config = ReflectiveMutationConfig.for_quick_experiments(
            minibatch_size=2,
            enable_detailed_logging=True,
            selection_temperature=0.8
        )
        
        # Overridden values
        assert config.minibatch_size == 2
        assert config.enable_detailed_logging is True
        assert config.selection_temperature == 0.8
        
        # Factory defaults that weren't overridden
        assert config.max_retries == 1
        assert config.module_selection_strategy == ModuleSelectionStrategy.RANDOM
        
        # Global defaults that weren't overridden
        assert config.preserve_original_on_failure is True

    def test_for_production_factory(self):
        """Test the for_production factory method."""
        config = ReflectiveMutationConfig.for_production()
        
        assert config.minibatch_size == 8
        assert config.max_retries == 5
        assert config.module_selection_strategy == ModuleSelectionStrategy.WORST_PERFORMING
        
        # Other defaults should remain unchanged
        assert config.selection_temperature == 1.0
        assert config.enable_detailed_logging is False

    def test_for_production_with_overrides(self):
        """Test the for_production factory with custom overrides."""
        config = ReflectiveMutationConfig.for_production(
            minibatch_size=12,
            selection_temperature=2.0,
            max_modules_per_generation=5
        )
        
        # Overridden values
        assert config.minibatch_size == 12
        assert config.selection_temperature == 2.0
        assert config.max_modules_per_generation == 5
        
        # Factory defaults that weren't overridden
        assert config.max_retries == 5
        assert config.module_selection_strategy == ModuleSelectionStrategy.WORST_PERFORMING
        
        # Global defaults that weren't overridden
        assert config.enable_detailed_logging is False

    def test_for_debugging_factory(self):
        """Test the for_debugging factory method."""
        config = ReflectiveMutationConfig.for_debugging()
        
        assert config.enable_detailed_logging is True
        assert config.minibatch_size == 2
        assert config.max_retries == 1
        assert config.preserve_original_on_failure is True
        
        # Global defaults that weren't overridden
        assert config.module_selection_strategy == ModuleSelectionStrategy.WORST_PERFORMING
        assert config.selection_temperature == 1.0

    def test_for_debugging_with_overrides(self):
        """Test the for_debugging factory with custom overrides."""
        config = ReflectiveMutationConfig.for_debugging(
            minibatch_size=1,
            max_retries=0,
            module_selection_strategy=ModuleSelectionStrategy.ALL
        )
        
        # Overridden values
        assert config.minibatch_size == 1
        assert config.max_retries == 0
        assert config.module_selection_strategy == ModuleSelectionStrategy.ALL
        
        # Factory defaults that weren't overridden
        assert config.enable_detailed_logging is True
        assert config.preserve_original_on_failure is True
        
        # Global defaults that weren't overridden
        assert config.selection_temperature == 1.0

    def test_factory_methods_validate_parameters(self):
        """Test that factory methods still validate parameters."""
        # Invalid parameters should still raise errors
        with pytest.raises(ValueError, match="minibatch_size must be positive"):
            ReflectiveMutationConfig.for_quick_experiments(minibatch_size=-1)
        
        with pytest.raises(ValueError, match="selection_temperature must be positive"):
            ReflectiveMutationConfig.for_production(selection_temperature=0.0)
        
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReflectiveMutationConfig.for_debugging(max_retries=-1)

    def test_config_immutability_after_creation(self):
        """Test that config values can be modified after creation (dataclass behavior)."""
        config = ReflectiveMutationConfig(minibatch_size=5)
        
        # Dataclasses are mutable by default
        config.minibatch_size = 10
        assert config.minibatch_size == 10
        
        # But direct validation won't run again
        config.minibatch_size = -1  # This won't raise an error
        assert config.minibatch_size == -1

    def test_config_with_all_strategy_types(self):
        """Test config creation with all different strategy types."""
        strategies = [
            ModuleSelectionStrategy.RANDOM,
            ModuleSelectionStrategy.WORST_PERFORMING,
            ModuleSelectionStrategy.ALL,
            ModuleSelectionStrategy.ROUND_ROBIN
        ]
        
        for strategy in strategies:
            config = ReflectiveMutationConfig(module_selection_strategy=strategy)
            assert config.module_selection_strategy == strategy

    def test_config_documentation_strings(self):
        """Test that configuration fields have proper documentation."""
        config = ReflectiveMutationConfig()
        
        # Test that the dataclass has proper field annotations
        annotations = ReflectiveMutationConfig.__annotations__
        
        assert 'minibatch_size' in annotations
        assert 'module_selection_strategy' in annotations
        assert 'max_retries' in annotations
        assert 'reflection_strategy' in annotations
        assert 'feedback_provider' in annotations
        assert 'enhanced_feedback_function' in annotations
        assert 'selection_temperature' in annotations
        assert 'max_modules_per_generation' in annotations
        assert 'enable_detailed_logging' in annotations
        assert 'preserve_original_on_failure' in annotations


class TestConfigurationIntegration:
    """Test integration scenarios with configuration objects."""

    def test_config_with_mock_dependencies(self):
        """Test configuration with mocked dependency objects."""
        mock_reflection = Mock()
        mock_feedback = Mock()
        mock_function = lambda x, y: (0.5, "test feedback")
        
        config = ReflectiveMutationConfig(
            reflection_strategy=mock_reflection,
            feedback_provider=mock_feedback,
            enhanced_feedback_function=mock_function
        )
        
        assert config.reflection_strategy == mock_reflection
        assert config.feedback_provider == mock_feedback
        assert config.enhanced_feedback_function == mock_function
        
        # Test that the function is callable
        result = config.enhanced_feedback_function("test", "test")
        assert result == (0.5, "test feedback")

    def test_config_serialization_behavior(self):
        """Test how configuration behaves with serialization-like operations."""
        config = ReflectiveMutationConfig(
            minibatch_size=7,
            module_selection_strategy=ModuleSelectionStrategy.ALL,
            enable_detailed_logging=True
        )
        
        # Test dict conversion-like behavior
        config_dict = {
            'minibatch_size': config.minibatch_size,
            'module_selection_strategy': config.module_selection_strategy,
            'enable_detailed_logging': config.enable_detailed_logging
        }
        
        assert config_dict['minibatch_size'] == 7
        assert config_dict['module_selection_strategy'] == ModuleSelectionStrategy.ALL
        assert config_dict['enable_detailed_logging'] is True

    def test_extreme_parameter_values(self):
        """Test configuration with extreme but valid parameter values."""
        # Very large values
        config = ReflectiveMutationConfig(
            minibatch_size=1000,
            max_retries=100,
            selection_temperature=100.0,
            max_modules_per_generation=1000
        )
        
        assert config.minibatch_size == 1000
        assert config.max_retries == 100
        assert config.selection_temperature == 100.0
        assert config.max_modules_per_generation == 1000
        
        # Very small but valid values
        config = ReflectiveMutationConfig(
            minibatch_size=1,
            max_retries=0,
            selection_temperature=0.001,
            max_modules_per_generation=1
        )
        
        assert config.minibatch_size == 1
        assert config.max_retries == 0
        assert config.selection_temperature == 0.001
        assert config.max_modules_per_generation == 1

    def test_factory_method_consistency(self):
        """Test that factory methods create consistent configurations."""
        # Multiple calls should create identical configs
        config1 = ReflectiveMutationConfig.for_quick_experiments()
        config2 = ReflectiveMutationConfig.for_quick_experiments()
        
        assert config1.minibatch_size == config2.minibatch_size
        assert config1.max_retries == config2.max_retries
        assert config1.module_selection_strategy == config2.module_selection_strategy
        
        # Different factory methods should create different configs
        quick_config = ReflectiveMutationConfig.for_quick_experiments()
        prod_config = ReflectiveMutationConfig.for_production()
        
        assert quick_config.minibatch_size != prod_config.minibatch_size
        assert quick_config.max_retries != prod_config.max_retries