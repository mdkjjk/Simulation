#量子テレポーテーション

import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt

import netsquid.components.instructions as instr
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression

class Example(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="example")
        self.num_runs = num_runs
        # Initialise sub-protocols
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=0,
                                           num_pairs=1, name="entangle_B"))
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))
        # Set start expressions
        self.subprotocols["entangle_A"].start_expression = self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING)
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(self.subprotocols["entangle_A"], Signals.SUCCESS)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i} Start")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
            resurl_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS, self)
            result = {
                "pos_A0": resurl_A["pos_A0"],
                "pos_A1": resurl_A["pos_A1"],
                "pos_B": signal_B,
                "time": sim_time() - start_time,
            }
            self.send_signal(Signals.SUCCESS, result)
            #print(f"Simulation {i} Finish")


class InitStateProgram(QuantumProgram):
    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_S, q1)
        yield self.run()


class BellMeasurement(NodeProtocol):
    def __init__(self, node, port, name=None):
        super().__init__(node, name)
        self.port = port
        self._qmem_pos0 = None
        self._qmem_pos1 = None

    def start(self):
        super().start()
        if self.start_expression is not None and not isinstance(self.start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(self.start_expression)))

    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        qubit_init_program = InitStateProgram()
        while True:
            #print(f"{self.name}: Start")
            expr_port = self.start_expression
            yield expr_port
            entanglement_ready = True
            source_protocol = expr_port.atomic_source
            ready_signal = source_protocol.get_signal_by_event(event=expr_port.triggered_events[0], receiver=self)
            self._qmem_pos1 = ready_signal.result
            #print(f"{self.name}: Entanglement received at {self._qmem_pos1}")
            while not self.node.qmemory.unused_positions:
                yield self.await_timer(1)
            self._qmem_pos0 = self.node.qmemory.unused_positions[0]
            self.node.qmemory.execute_program(qubit_init_program, qubit_mapping=[self._qmem_pos0])
            expr_signal = self.await_program(self.node.qmemory)
            yield expr_signal
            qubit_initialised = True
            #print(f"{self.name}: Initqubit received at {self._qmem_pos0}")
            if qubit_initialised and entanglement_ready:
                self.node.qmemory.operate(ns.CNOT, [self._qmem_pos0, self._qmem_pos1])
                self.node.qmemory.operate(ns.H, self._qmem_pos0)
                m, _ = self.node.qmemory.measure([self._qmem_pos0, self._qmem_pos1])
                # Send measurement results to Bob:
                self.port.tx_output(m)
                result = {"pos_A0": self._qmem_pos0,
                          "pos_A1": self._qmem_pos1,}
                self.send_signal(Signals.SUCCESS, result)
                #print(f"{self.name}: Finish")
                qubit_initialised = False
                entanglement_ready = False


class Correction(NodeProtocol):
    def __init__(self, node, name=None):
        super().__init__(node, name)
        self._qmem_pos = None

    def run(self):
        port_alice = self.node.ports["cin_alice"]
        port_charlie = self.node.ports["qin_charlie"]
        entanglement_ready = False
        meas_results = None
        while True:
            #print(f"{self.name}: Start")
            evexpr_port_a = self.await_port_input(port_alice)
            evexpr_port_c = self.await_port_input(port_charlie)
            expression = yield evexpr_port_a | evexpr_port_c
            if expression.first_term.value:
                meas_results = port_alice.rx_input().items
                #print(f"{self.name}: Result received")
            else:
                entanglement_ready = True
                #print(f"{self.name}: Entanglement received")
            if meas_results is not None and entanglement_ready:
                # Do corrections (blocking)
                if meas_results[0] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_Z)
                if meas_results[1] == 0:
                    self.node.qmemory.execute_instruction(instr.INSTR_X)
                self.send_signal(Signals.SUCCESS, 0)
                #print(f"{self.name}: Teleport success")
                entanglement_ready = False
                meas_results = None


def example_network_setup(source_delay=1e5, source_fidelity_sq=1.0, depolar_rate=2000,
                          node_distance=30):
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    conn_cchannel = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel, port_name_node1="cout_bob", port_name_node2="cin_alice")
    # node_A.connect_to(node_B, conn_cchannel)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate),
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charlie")
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network


def example_sim_setup(node_a, node_b, num_runs):
    example = Example(node_a, node_b, num_runs=num_runs)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        node_a.qmemory.pop(positions=[result["pos_A0"]])
        node_a.qmemory.pop(positions=[result["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        return {"F2": f2, "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=example,
                                     event_type=Signals.SUCCESS.value))
    return example, dc


def run_experiment(node_distances):
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        ns.sim_reset()
        network = example_network_setup(node_distance=node_distance)
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        example, dc = example_sim_setup(node_a, node_b, 1000)
        example.start()
        ns.sim_run()
        df = dc.dataframe
        df['node_distance'] = node_distance
        fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data


def create_plot():
    matplotlib.use('Agg')
    node_distances = [1 + i for i in range(0, 100, 5)]
    fidelities = run_experiment(node_distances)
    plot_style = {'kind': 'scatter', 'grid': True,
                  'title': "Fidelity of the teleported quantum state"}
    data = fidelities.groupby("node_distance")['F2'].agg(
        fidelity='mean', sem='sem').reset_index()
    save_dir = "./plots_2000"
    existing_files = len([f for f in os.listdir(save_dir) if f.startswith("Original_Teleportation")])
    filename = f"{save_dir}/Original_Teleportation fidelity_{existing_files + 1}.png"
    data.plot(x='node_distance', y='fidelity', yerr='sem', **plot_style)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    fidelities.to_csv(f"{save_dir}/Original_Teleportation fidelity_{existing_files + 2}.csv")


if __name__ == "__main__":
    #network = example_network_setup()
    #example, dc = example_sim_setup(network.get_node("node_A"),network.get_node("node_B"),num_runs=1000)
    #example.start()
    #ns.sim_run()
    #print("Average fidelity of received qubit: {}".format(dc.dataframe["F2"].mean()))
    create_plot()
