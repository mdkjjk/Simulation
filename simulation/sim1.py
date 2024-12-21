import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
import netsquid.components.instructions as instr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
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
        self.add_subprotocol(BellMeasurementProtocol(node=node_a, name="teleportation_A", qubit_protocol=InitStateProtocol(node_a)))
        self.add_subprotocol(CorrectionProtocol(node=node_b, name="teleportation_B"))
        # Set start expressions
        self.subprotocols["entangle_A"].start_expression = self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING)
        self.subprotocols["teleportation_A"].start_expression = (
            self.subprotocols["teleportation_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["teleportation_B"].start_expression = (
            self.subprotocols["teleportation_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))

    def run(self):
        print(f"Starting {self.name} protocol.")
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["entangle_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["entangle_B"], Signals.SUCCESS))
            print(f"Received signal {Signals.SUCCESS} in {self.name} protocol at time {sim_time()}")
            mem_pos_a0 = self.subprotocols["entangle_A"].get_signal_result(Signals.SUCCESS, self)
            mem_pos_b = self.subprotocols["entangle_B"].get_signal_result(Signals.SUCCESS, self)
            self.subprotocols["teleportation_A"].mem_pos1 = mem_pos_a0
            self.subprotocols["teleportation_B"].mem_pos = mem_pos_b
            yield self.await_signal(self.subprotocols["teleportation_B"], Signals.SUCCESS)
            fidelity = self.subprotocols["teleportation_B"].get_signal_result(Signals.SUCCESS, self)
            self.send_signal(Signals.SUCCESS, fidelity)
            print(f"Signal SUCCESS sent with fidelity: {fidelity}")

class InitStateProtocol(NodeProtocol): #@Alice
    def run(self):
        print(f"Starting {self.name} protocol.")
        qubit, = qapi.create_qubits(1)
        mem_pos = self.node.qmemory.unused_positions[0]
        self.node.qmemory.put(qubit, mem_pos)
        self.node.qmemory.operate(ns.H, mem_pos)
        self.node.qmemory.operate(ns.S, mem_pos)
        self.send_signal(signal_label=Signals.SUCCESS, result=mem_pos)
        print(f"Signal {Signals.SUCCESS} sent with result: {mem_pos} at time {sim_time()}")

#Aliceの持つ2つの量子ビットを測定
class BellMeasurementProtocol(NodeProtocol): #@Alice
    def __init__(self, node, qubit_protocol, start_expression=None, mem_pos0=None, mem_pos1=None, name=None):
        super().__init__(node=node, name=name)
        self.add_subprotocol(qubit_protocol, 'initprotocol')
        self.start_expression = start_expression
        self.mem_pos0 = mem_pos0
        self.mem_pos1 = mem_pos1

    def run(self):
        print(f"Starting {self.name} protocol.")
        qubit_initialised = False
        entanglement_ready = False
        while True:
            evexpr_signal = self.await_signal(sender=self.subprotocols['initprotocol'], signal_label=Signals.SUCCESS)
            evexpr_port = self.await_port_input(self.node.ports["qin_charlie"]) #ポートがメッセージを受け取るまで待機
            expression = yield evexpr_signal | evexpr_port
            print(f"Received signal {Signals.SUCCESS} in {self.name} protocol at time {sim_time()}")
            if expression.first_term.value:
                 # First expression was triggered
                qubit_initialised = True #シグナルが送信されたら、量子ビットの生成完了
                self.mem_pos0 = self.subprotocols["initprotocol"].get_signal_result(Signals.SUCCESS, self)
            else:
                # Second expression was triggered
                entanglement_ready = True
            if qubit_initialised and entanglement_ready:
                # Perform Bell measurement:
                self.node.qmemory.operate(ns.CNOT, [self.mem_pos0, self.mem_pos1])
                self.node.qmemory.operate(ns.H, self.mem_pos0)
                m, _ = self.node.qmemory.measure([self.mem_pos0, self.mem_pos1])
                self.node.qmemory.pop(position=[self.mem_pos0])
                self.node.qmemory.pop(position=[self.mem_pos1])
                # Send measurement results to Bob:
                self.node.ports["cout_bob"].tx_output(m) #測定結果を送信
                self.send_signal(Signals.SUCCESS)
                print(f"Signal {Signals.SUCCESS} sent at time {sim_time()}")
                qubit_initialised = False
                entanglement_ready = False

    def start(self):
        super().start()
        self.start_subprotocols()

class CorrectionProtocol(NodeProtocol): #@Bob
    def __init__(self, node, start_expression=None, mem_pos=None, name=None):
        super().__init__(node=node, name=name)
        self.start_expression = start_expression
        self.mem_pos = mem_pos

    def run(self):
        print(f"Starting {self.name} protocol.")
        port_alice = self.node.ports["cin_alice"]
        port_charlie = self.node.ports["qin_charlie"]
        entanglement_ready = False
        meas_results = None
        while True:
            evexpr_port_a = self.await_port_input(port_alice)
            evexpr_port_c = self.start_expression
            expression = yield evexpr_port_a | evexpr_port_c
            if expression.first_term.value:
                meas_results = port_alice.rx_input().items #測定結果を受け取る
            else:
                entanglement_ready = True #エンタングルビットを受け取る
            if meas_results is not None and entanglement_ready:
                if meas_results[0]:
                    self.node.qmemory.operate(ns.Z, self.mem_pos)
                if meas_results[1]:
                    self.node.qmemory.operate(ns.X, self.mem_pos)
                fidelity = ns.qubits.fidelity(self.node.qmemory.peek(self.mem_pos)[0],
                                              ns.y0, squared=True)
                self.node.qmemory.pop(position=[self.mem_pos])
                self.send_signal(Signals.SUCCESS, fidelity)
                entanglement_ready = False
                meas_results = None

def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1000,
                          node_distance=20):
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    conn_cchannel = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel,
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # node_A.connect_to(node_B, conn_cchannel)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_loss_model": None,
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum",
        port_name_node1="qin_charlie", port_name_node2="qin_charlie")
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
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}

    def collect_fidelity_data(evexpr):
        protocol = evexpr.triggered_events[-1].source
        fidelity = protocol.get_signal_result(Signals.SUCCESS)
        print(fidelity)
        return {"fidelity": fidelity}

    dc = DataCollector(collect_fidelity_data, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=example,
                                     event_type=Signals.SUCCESS.value))
    return example, dc


if __name__ == "__main__":
    network = example_network_setup()
    example, dc = example_sim_setup(network.get_node("node_A"),
                                         network.get_node("node_B"),
                                         num_runs=10)
    example.start()
    ns.sim_run()
    print("Average fidelity of generated entanglement: {}".format(dc.dataframe.mean()))
