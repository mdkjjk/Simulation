import netsquid as ns
import netsquid.components.instructions as instr
from netsquid.components.qmemory import QuantumMemory

#Instructions for quantum memory devices
qmemory = QuantumMemory('ExampleQMem', num_positions=1)
instr.INSTR_INIT(qmemory, positions=[0]) #初期化|0>
instr.INSTR_H(qmemory, positions=[0]) #ハダマートゲートを適用
print(instr.INSTR_MEASURE_X(qmemory, positions=[0])) #ハダマート基底で測定

from netsquid.qubits import operators as ops
INSTR_XY = instr.IGate("xy_gate", ops.X * ops.Y) #オリジナルのゲートを作成

from netsquid.components.qprocessor import QuantumProcessor

qproc = QuantumProcessor("ExampleQPU", num_positions=3,
                         fallback_to_nonphysical=True)
qproc.execute_instruction(instr.INSTR_INIT, [0, 1])
qproc.execute_instruction(instr.INSTR_H, [1])
qproc.execute_instruction(instr.INSTR_CNOT, [1, 0])
m1 = qproc.execute_instruction(instr.INSTR_MEASURE, [0])
m2 = qproc.execute_instruction(instr.INSTR_MEASURE, [1])
print(m1 == m2)  # Measurement results are both the same (either both 1 or both 0)
print(ns.sim_time())

#Physical instructions
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction

phys_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=3),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True,
                        topology=[(0, 1), (2, 1)]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(
        instr.INSTR_MEASURE, duration=7, parallel=False,
        quantum_noise_model=DepolarNoiseModel(depolar_rate=0.01, time_independent=True),
        apply_q_noise_after=False, topology=[1]),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=True,
                        topology=[0, 2])
]
noisy_qproc = QuantumProcessor("NoisyQPU", num_positions=3,
                               mem_noise_models=[DepolarNoiseModel(1e7)] * 3,
                               phys_instructions=phys_instructions)

ns.sim_time()
noisy_qproc.execute_instruction(instr.INSTR_INIT, [0, 1])
ns.sim_run()
ns.sim_time()

#Quantum programs
from netsquid.components.qprogram import QuantumProgram

prog = QuantumProgram(num_qubits=2)
q1, q2 = prog.get_qubit_indices(2)  # Get the qubit indices we'll be working with
prog.apply(instr.INSTR_INIT, [q1, q2])
prog.apply(instr.INSTR_H, q1)
prog.apply(instr.INSTR_CNOT, [q1, q2])
prog.apply(instr.INSTR_MEASURE, q1, output_key="m1")
prog.apply(instr.INSTR_MEASURE, q2, output_key="m2")

noisy_qproc.reset()
ns.sim_reset()
noisy_qproc.execute_program(prog, qubit_mapping = [2, 1])
ns.sim_run()
print(ns.sim_time())
print(prog.output["m1"] == prog.output["m2"])  
print(prog.output["m1"], prog.output["m2"])

class EntangleProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_INIT, [q1, q2])
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_MEASURE, q1, output_key="m1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="m2")
        yield self.run()

class ControlledQProgram(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="m1")
        yield self.run()
        # Depending on outcome on q1 either flip q2 or q3
        if self.output["m1"][0] == 0:
            self.apply(instr.INSTR_X, q2)
        else:
            self.apply(instr.INSTR_X, q3)
        self.apply(instr.INSTR_MEASURE, q2, output_key="m2")
        self.apply(instr.INSTR_MEASURE, q3, output_key="m3")
        yield self.run(parallel=False)

#Local teleportation example using programs
noisy_qproc.reset()
ns.sim_reset()
ns.set_qstate_formalism(ns.QFormalism.DM)

class TeleportationProgram(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        q0, q1, q2 = self.get_qubit_indices(3)
        # Entangle q1 and q2:
        self.apply(instr.INSTR_INIT, [q0, q1, q2])
        self.apply(instr.INSTR_H, q2)
        self.apply(instr.INSTR_CNOT, [q2, q1])
        # Set q0 to the desired state to be teleported:
        self.apply(instr.INSTR_H, q0)
        self.apply(instr.INSTR_S, q0)
        # Bell measurement:
        self.apply(instr.INSTR_CNOT, [q0, q1])
        self.apply(instr.INSTR_H, q0)
        self.apply(instr.INSTR_MEASURE, q0, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q1, output_key="M2")
        yield self.run()
        # Do corrections:
        if self.output["M2"][0] == 1:
            self.apply(instr.INSTR_X, q2)
        if self.output["M1"][0] == 1:
            self.apply(instr.INSTR_Z, q2)
        yield self.run()


noisy_qproc.execute_program(TeleportationProgram())
ns.sim_run()
qubit = noisy_qproc.pop(2)
fidelity = ns.qubits.fidelity(
    qubit, ns.qubits.outerprod((ns.S*ns.H*ns.s0).arr), squared=True)
print(f"{fidelity:.3f}")

#Additional features
class CheatingQProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_X, q1)
        self.apply(instr.INSTR_SIGNAL, physical=False)
        self.apply(instr.INSTR_Z, q1, physical=False)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_MEASURE, q1, output_key="m1", physical=False)
        self.apply(instr.INSTR_MEASURE, q2, output_key="m2", physical=False)
        yield self.run()

class LoadingQProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        # Run a regular sequence
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_X, q1)
        yield self.run()
        # Load and run another program
        yield from self.load(CheatingQProgram)
