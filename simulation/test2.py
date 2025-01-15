import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression


class Filter(NodeProtocol):
    def __init__(self, node, port, start_expression=None, msg_header="filter",
                 epsilon=0.3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "Filter({}, {})".format(node.name, port.name)
        super().__init__(node, name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_OK = False
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_measurement_operators(epsilon)

    def _set_measurement_operators(self, epsilon):
        m0 = ops.Operator("M0", np.sqrt(epsilon) * outerprod(s0) + outerprod(s1))
        m1 = ops.Operator("M1", np.sqrt(1 - epsilon) * outerprod(s0))
        self.meas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                #print(f"{self.name}: Start II")
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                #print(f"{self.name}: Start I")
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                self._qmem_pos = ready_signal.result
                yield from self._handle_qubit_rx()

    # TODO does start reset vars?
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_OK = False
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        # Handle incoming Qubit on this node.
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # Retrieve Qubit from input store
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        m = output["instr"][0]
        # m = INSTR_MEASURE(self.node.qmemory, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        self.local_qcount += 1
        self.local_meas_OK = (m == 0)
        self.port.tx_output(Message([self.local_qcount, self.local_meas_OK], header=self.header))
        self._check_success()

    def _handle_cchannel_rx(self):
        # Handle incoming classical message from sister node.
        if (self.local_qcount == self.remote_qcount and
                self._qmem_pos is not None and
                self.node.qmemory.mem_positions[self._qmem_pos].in_use):
            self._check_success()

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            # SUCCESS!
            self.send_signal(Signals.SUCCESS, self._qmem_pos)
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status
            pass
        else:
            # FAILURE
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)

    def _handle_fail(self):
        if self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 1:
            return False
        return True


class FilteringExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, epsilon=0.9):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Filtering example")
        self._epsilon = epsilon
        self.num_runs = num_runs
        # Initialise sub-protocols
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A"))
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=1,
                          name="entangle_B"))
        self.add_subprotocol(Filter(node_a, node_a.get_conn_port(node_b.ID),
                                    epsilon=epsilon, name="purify_A"))
        self.add_subprotocol(Filter(node_b, node_b.get_conn_port(node_a.ID),
                                    epsilon=epsilon, name="purify_B"))
        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)


def example_network_setup(source_delay=1e5, source_fidelity_sq=1.0, depolar_rate=2000,
                          node_distance=30):
    network = Network("purify_network")

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
    network.add_connection(node_a, node_b, connection=conn_cchannel)
    # node_A.connect_to(node_B, conn_cchannel)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate),
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum")
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network


def example_sim_setup(node_a, node_b, num_runs):
    filt_example = FilteringExample(node_a, node_b, num_runs=num_runs, epsilon=0.9)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=filt_example,
                                     event_type=Signals.SUCCESS.value))
    return filt_example, dc


def run_experiment(node_distances):
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        ns.sim_reset()
        network = example_network_setup(node_distance=node_distance)
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        example, dc = example_sim_setup(node_a, node_b, 100)
        example.start()
        ns.sim_run()
        df = dc.dataframe
        df['node_distance'] = node_distance
        fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data


def create_plot():
    matplotlib.use('Agg')
    node_distances = [i for i in range(5, 100, 5)]
    fidelities = run_experiment(node_distances)
    plot_style = {'kind': 'scatter', 'grid': True,
                  'title': "Fidelity of entanglement with filtering"}
    data = fidelities.groupby("node_distance")['F2'].agg(
        fidelity='mean', sem='sem').reset_index()
    save_dir = "./plots"
    existing_files = len([f for f in os.listdir(save_dir) if f.startswith("Filtering_Entanglement")])
    filename = f"{save_dir}/Filtering_Entanglement fidelity_{existing_files + 1}.png"
    data.plot(x='node_distance', y='fidelity', yerr='sem', **plot_style)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    fidelities.to_csv(f"{save_dir}/Filtering_Entanglement fidelity_{existing_files + 2}.csv")


if __name__ == "__main__":
    #network = example_network_setup()
    #filt_example, dc = example_sim_setup(network.get_node("node_A"),network.get_node("node_B"),num_runs=2)
    #filt_example.start()
    #ns.sim_run()
    #print("Average fidelity of generated entanglement with filtering: {}".format(dc.dataframe["F2"].mean()))
    create_plot()