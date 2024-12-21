import netsquid as ns
import numpy as np
import pydynaa
from netsquid.components.models.qerrormodels import DephaseNoiseModel, DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction
import netsquid.components.instructions as instr
from netsquid.components.qprocessor import QuantumProcessor

def create_qprocessor(name):
    noise_rate = 200
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_X, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(instr.INSTR_Z, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False,
                             mem_noise_models=[mem_noise_model] * 2,
                             phys_instructions=physical_instructions)
    return qproc

from netsquid.nodes.network import Network
from netsquid.nodes import Node
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection

def setup_network(num_nodes, node_distance, source_frequency):
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    network = Network("Repeater_chain_network")
    # Create nodes with quantum processors
    nodes = []
    for i in range(num_nodes):
        # Prepend leading zeros to the number
        num_zeros = int(np.log10(num_nodes)) + 1
        nodes.append(Node(f"Node_{i:0{num_zeros}d}", qmemory=create_qprocessor(f"qproc_{i}")))
    network.add_nodes(nodes)
    # Create quantum and classical connections:
    for i in range(num_nodes - 1):
        node, node_right = nodes[i], nodes[i + 1]
        # Create quantum connection
        qconn = EntanglingConnection(name=f"qconn_{i}-{i+1}", length=node_distance,
                                     source_frequency=source_frequency)
        # Add a noise model which depolarizes the qubits exponentially
        # depending on the connection length
        for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
            qconn.subcomponents[channel_name].models['quantum_noise_model'] =\
                FibreDepolarizeModel()
        port_name, port_r_name = network.add_connection(
            node, node_right, connection=qconn, label="quantum")
        # Forward qconn directly to quantum memories for right and left inputs:
        node.ports[port_name].forward_input(node.qmemory.ports["qin0"])  # R input
        node_right.ports[port_r_name].forward_input(
            node_right.qmemory.ports["qin1"])  # L input
        # Create classical connection
        cconn = ClassicalConnection(name=f"cconn_{i}-{i+1}", length=node_distance)
        port_name, port_r_name = network.add_connection(
            node, node_right, connection=cconn, label="classical",
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        # Forward cconn to right most node
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(
                lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))
    return network

from netsquid.components.models.qerrormodels import QuantumErrorModel
class FibreDepolarizeModel(QuantumErrorModel):
    def __init__(self, p_depol_init=0.009, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
                10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)

from netsquid.protocols.nodeprotocols import LocalProtocol
def setup_repeater_protocol(network):
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    for node in nodes[1:-1]:
        subprotocol = SwapProtocol(node=node, name=f"Swap_{node.name}")
        protocol.add_subprotocol(subprotocol)
    # Add CorrectProtocol to Bob
    subprotocol = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(subprotocol)
    return protocol

from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.component import Message
class SwapProtocol(NodeProtocol):
    def __init__(self, node, name):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(instr.INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
            m, = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))

import netsquid.qubits.ketstates as ks
class CorrectProtocol(NodeProtocol):
    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["ccon_L"])
            message = self.node.ports["ccon_L"].rx_input()
            if message is None or len(message.items) != 1:
                continue
            m = message.items[0]
            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr += 1
            self._counter += 1
            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0

from netsquid.components.qprogram import QuantumProgram
class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(instr.INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(instr.INSTR_Z, q1)
        yield self.run()

from netsquid.util.datacollector import DataCollector
from netsquid.protocols import NodeProtocol, Signals
def setup_datacollector(network, protocol):
    # Ensure nodes are ordered in the chain:
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]

    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([0])
        qubit_b, = nodes[-1].qmemory.peek([1])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        return {"fidelity": fidelity}

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    dc.collect_on(pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value))
    return dc

def run_simulation(num_nodes=4, node_distance=20, num_iters=100):
    ns.sim_reset()
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    network = setup_network(num_nodes, node_distance=node_distance,
                            source_frequency=1e9 / est_runtime)
    protocol = setup_repeater_protocol(network)
    dc = setup_datacollector(network, protocol)
    protocol.start()
    ns.sim_run(est_runtime * num_iters)
    return dc.dataframe

import pandas, os
def create_plot(num_iters=2000):
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    for distance in [10, 30, 50]:
        data = pandas.DataFrame()
        for num_node in range(3, 20):
            data[num_node] = run_simulation(num_nodes=num_node,
                                            node_distance=distance / num_node,
                                            num_iters=num_iters)['fidelity']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        save_dir = "./plots"
        existing_files = len([f for f in os.listdir(save_dir) if f.startswith("Repeater")])
        filename = f"{save_dir}/Repeater_{existing_files + 1}.png"
        data.plot(y='fidelity', yerr='sem', label=f"{distance} km", ax=ax)
    plt.xlabel("number of nodes")
    plt.ylabel("fidelity")
    plt.title("Repeater chain with different total lengths")
    plt.savefig(filename)

print("This example module is located at: {}".format(
      ns.examples.repeater_chain.__file__))
create_plot()