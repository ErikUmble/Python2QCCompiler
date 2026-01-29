import benchmarklib
from benchmarklib import BenchmarkDatabase, CompileType
from benchmarklib.algorithms import GroverRunner, GroverConfig, calculate_grover_iterations
from benchmarklib.pipeline import PipelineCompiler
from benchmarklib.pipeline.synthesis import QuantumMPC, XAGSynthesizer, QCFSynthesizer
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager

service = QiskitRuntimeService()
backend = service.backend(name="ibm_rensselaer")

compiler_qmpc_ai_test = PipelineCompiler(
    synthesizer = QuantumMPC(),
    steps = [], 
    backend = backend, 
    pass_manager_factory = generate_ai_pass_manager,
    transpile_options = {"ai_optimization_level": 3, "optimization_level": 3}
)

compiler_qmpc_standard = PipelineCompiler(
    synthesizer = QuantumMPC(),
    steps = [], 
    name="QuantumMPC",  # for backwards compatibility
    backend = backend, 
    pass_manager_factory = generate_preset_pass_manager,
    transpile_options = {"optimization_level": 3}
)

compiler_qcf = PipelineCompiler(
    synthesizer = QCFSynthesizer(),
    steps = [], 
    name="CLASSICAL_FUNCTION",  # for backwards compatibility
    backend = backend, 
    pass_manager_factory = generate_preset_pass_manager,
    transpile_options = {"optimization_level": 3}
)

compiler_xag = PipelineCompiler(
    synthesizer = XAGSynthesizer(),
    steps = [], 
    name="XAG",  # for backwards compatibility
    backend = backend, 
    pass_manager_factory = generate_preset_pass_manager,
    transpile_options = {"optimization_level": 3}
)