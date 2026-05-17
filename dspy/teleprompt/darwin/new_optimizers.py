from copy import Error
from typing import Union, Optional, Dict, Any, Type, ForwardRef, List, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect, uuid, re
from enum import Enum

def pascal_case(name:str):
    # Remove non-alphanumeric characters
    name = re.sub(r'\W+', '', name)
    # Ensure the name starts with a letter
    while len(name) > 3 and name[0].isalpha():
        name = name[1:]
    if len(name) < 3:
        raise ValueError("Class name must start with a letter and be greater than 3 caracters")
    return ''.join(part.lower().capitalize() for part in name.split())

def snake_case(name:str) -> str:
    s = name.replace(' ', '_')
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

@dataclass
class WorkflowLoop:
    name: str
    condition: Condition  # Condition to continue looping
    body: Union[str, 'Optimizer']  # Workflow to execute in loop
    max_iterations: int = 100
    loop_variable: Optional[str] = None  # Variable to store iteration count
    break_condition: Optional[Condition] = None  # Optional early break condition
    description: str = ""


@dataclass
class WorkflowBranch:
    name: str
    condition: Condition  # Condition that evaluates to branch key
    branches: Dict[str, Union[str, 'Optimizer']]  # Branch key -> workflow reference
    default_branch: Optional[str] = None  # Default if no branch matches
    description: str = ""
class ConditionType(Enum):
    FUNCTION = "function"
    EXPRESSION = "expression"
    METHOD = "method"

@dataclass
class Condition:
    name: str
    condition_type: ConditionType
    condition_ref: Union[str, Type, ForwardRef, Callable]  # Function, method, or expression
    args_schema: Dict[str, str] = field(default_factory=dict)  # For AI agents
    description: str = ""  # Human/AI readable description

@dataclass
class WorkflowNode:
    _id: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))
    _workflow: 'Workflow'
    @property
    def id(self):
        return self._id
    @property
    def workflow(self):
        return self._workflow

    def __hash__(self) -> int:
        """Hash based on unique ID"""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on unique ID"""
        return isinstance(other, WorkflowNode) and (self.id == other.id)

@dataclass
class NamedNode(WorkflowNode):
    _name: str
    desc: str = ""

    def __post_init__(self):
       self._ref = snake_case(self._name)

    @property
    def ref(self):
        return self._ref

    @property
    def name(self):
        return self._name



# Workflow data structures
@dataclass
class Workflow(NamedNode):
    # This field will be set to self in __post_init__
    _workflow: 'Workflow' = field(init=False, repr=False)  # repr=False to avoid infinite recursion
    # Additional workflow-specific fields
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)
    entry_points: Dict[str, WorkflowNode] = field(default_factory=dict)
    _last_item: WorkflowNode | None = None

    def __post_init__(self):
        """Set the workflow field to reference itself"""
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, 'workflow', self)
        # Also set the workflow field in the base class part
        object.__setattr__(self, '_base_workflow', self)

    def add_edge(self, source: 'WorkflowNode', node: 'WorkflowNode') -> 'WorkflowNode':
        self.nodes[source.id] = source
        self.nodes[node.id] = node
        self.edges[source.id] = node.id
        return node

    def entry_point(self, name: str, *args, **kwargs) -> 'WorkflowNode':
        node = NamedNode(self, name, *args, **kwargs)
        self.entry_points[node.id] = node
        self._last_item = node
        return self._last_item

    def then(self, name:str, *args, **kwargs) -> 'WorkflowNode':
        if self._last_item == None:
            return self.entry_point(name, *args, **kwargs)
        node = WorkflowStep(self, name, *args, **kwargs)
        self._last_item = self.add_edge(self._last_item, node)
        return self._last_item


@dataclass
class WorkflowStep(NamedNode):
    inherit: Union[str, Type, ForwardRef] = ""
    method: str = ""
    args: Dict[str, str] = field(default_factory=dict)
    returns: str = ""
    returns_ref: str = ""
    errors: List[Error|Exception] = field(default_factory=list)

@dataclass
class WorkflowBlock(WorkflowNode):
    condition: Union[str, Type, ForwardRef, Callable, None]  # Function, method, or expression
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)
    entry_points: Dict[str, WorkflowNode] = field(default_factory=dict)


class Ecosystem:
    def __init__(self, *args, **kwargs):
        self.optimizer = None
    def load_optimizer(self, optimizer:'Optimizer'):
        self.optimizer = optimizer
        ... # load values into the optimizer

class Optimizer:
    def __new__(cls, name:str, desc:str="") -> Type["Optimizer"]:
        def __new__(cls, ecosystem:Ecosystem):
            return object.__new__(cls)
        def __init__(self, ecosystem:Ecosystem):
            ecosystem.load_optimizer(self)

        class_dict = {
            "__new__": __new__,
            "__init__": __init__
        }
        opt_cls = type(name, (cls,), class_dict)
        opt_cls.__doc__ = desc
        setattr(opt_cls, 'workflow', Workflow(name, desc=desc))
        setattr(opt_cls, '_last_item', None)


        def branch(cls, condition_stage: DarwinStage, branches: Dict[str, 'DarwinWorkflow']):
            """Add branching logic"""
            cls.stages.append(('branch', condition_stage, branches))
            return cls

        def loop(cls, condition_stage: DarwinStage, body_workflow: 'DarwinWorkflow',
            max_iterations: int = 100):
            """Add loop with condition"""
            cls.stages.append(('loop', condition_stage, body_workflow, max_iterations))
            return cls

        def parallel(self, workflows: List['DarwinWorkflow']):
            """Execute workflows in parallel"""
            cls.stages.append(('parallel', workflows))
            return cls

        return opt_cls




GEPA = Optimizer('GEPA', desc="""Create a Darwin optimizer using composite workflow architecture.

This is the core GEPA factory that assembles a complete composite workflow
from specialized components and phases.

Args:
    generator: Generation component (e.g., ReflectivePromptMutation)
    metric: Evaluation metric function
    budget: Budget component for LLM call management
    minibatch_size: Size of minibatches for evaluation
    patience: Generations without progress before termination
    verbose: Enable detailed logging

Returns:
    Darwin orchestrator with GEPA composite workflow
""")
eco = Ecosystem()
gepa = GEPA(eco)
print(f"gepa is instance of Optimizer:{isinstance(gepa, Optimizer)} - Object: {gepa}")
print(f"Ecosystem contains gepa: {eco.optimizer == gepa}")
