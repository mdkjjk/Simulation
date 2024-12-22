from netsquid.protocols import Protocol
import netsquid as ns

#Protocol
class WaitProtocol(Protocol):
    def run(self):
        print(f"Starting protocol at {ns.sim_time()}")
        yield self.await_timer(100)
        print(f"Ending protocol at {ns.sim_time()}")

ns.sim_reset()
protocol = WaitProtocol()
protocol.start()
stats = ns.sim_run()

from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi

#The ping pong example using protocols
class PingProtocol(NodeProtocol):
    def run(self):
        print(f"Starting ping at t={ns.sim_time()}")
        port = self.node.ports["port_to_channel"]
        qubit, = qapi.create_qubits(1)
        port.tx_output(qubit)  # Send qubit to Pong
        while True:
            # Wait for qubit to be received back
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.Z)
            labels_z =  ("|0>", "|1>")
            print(f"{ns.sim_time()}: Pong event! {self.node.name} measured "
                  f"{labels_z[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # Send qubit to B


class PongProtocol(NodeProtocol):
    def run(self):
        print("Starting pong at t={}".format(ns.sim_time()))
        port = self.node.ports["port_to_channel"]
        while True:
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.X)
            labels_x = ("|+>", "|->")
            print(f"{ns.sim_time()}: Ping event! {self.node.name} measured "
                  f"{labels_x[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # send qubit to Ping

ns.sim_reset()
ns.set_random_state(seed=42)
node_ping = Node("Ping", port_names=["port_to_channel"])
node_pong = Node("Pong", port_names=["port_to_channel"])
connection = DirectConnection("Connection",
                              QuantumChannel("Channel_LR", delay=10),
                              QuantumChannel("Channel_RL", delay=10))
node_ping.ports["port_to_channel"].connect(connection.ports["A"])
node_pong.ports["port_to_channel"].connect(connection.ports["B"])
ping_protocol = PingProtocol(node_ping)
pong_protocol = PongProtocol(node_pong)

ping_protocol.start()
pong_protocol.start()
stats = ns.sim_run(91)

pong_protocol.stop()
stats = ns.sim_run()  

pong_protocol.start()
stats = ns.sim_run()

ping_protocol.reset()
stats = ns.sim_run(duration=51)

#The teleportation example using protocols
from netsquid.protocols import NodeProtocol, Signals

class InitStateProtocol(NodeProtocol):
    def run(self):
        qubit, = qapi.create_qubits(1)
        mem_pos = self.node.qmemory.unused_positions[0]
        self.node.qmemory.put(qubit, mem_pos)
        self.node.qmemory.operate(ns.H, mem_pos)
        self.node.qmemory.operate(ns.S, mem_pos)
        self.send_signal(signal_label=Signals.SUCCESS, result=mem_pos)


from pydynaa import EventExpression

class BellMeasurementProtocol(NodeProtocol):
    def __init__(self, node, qubit_protocol):
        super().__init__(node)
        self.add_subprotocol(qubit_protocol, 'qprotocol')

    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        while True:
            evexpr_signal = self.await_signal(
                sender=self.subprotocols['qprotocol'],
                signal_label=Signals.SUCCESS)
            evexpr_port = self.await_port_input(self.node.ports["qin_charlie"])
            expression = yield evexpr_signal | evexpr_port
            if expression.first_term.value:
                 # First expression was triggered
                qubit_initialised = True
            else:
                # Second expression was triggered
                entanglement_ready = True
            if qubit_initialised and entanglement_ready:
                # Perform Bell measurement:
                self.node.qmemory.operate(ns.CNOT, [0, 1])
                self.node.qmemory.operate(ns.H, 0)
                m, _ = self.node.qmemory.measure([0, 1])
                # Send measurement results to Bob:
                self.node.ports["cout_bob"].tx_output(m)
                self.send_signal(Signals.SUCCESS)
                print(f"{ns.sim_time():.1f}: Alice received entangled qubit, "
                      f"measured qubits & sending corrections")
                break

    def start(self):
        super().start()
        self.start_subprotocols()

class CorrectionProtocol(NodeProtocol):

    def __init__(self, node):
        super().__init__(node)

    def run(self):
        port_alice = self.node.ports["cin_alice"]
        port_charlie = self.node.ports["qin_charlie"]
        entanglement_ready = False
        meas_results = None
        while True:
            evexpr_port_a = self.await_port_input(port_alice)
            evexpr_port_c = self.await_port_input(port_charlie)
            expression = yield evexpr_port_a | evexpr_port_c
            if expression.first_term.value:
                meas_results = port_alice.rx_input().items
            else:
                entanglement_ready = True
            if meas_results is not None and entanglement_ready:
                if meas_results[0]:
                    self.node.qmemory.operate(ns.Z, 0)
                if meas_results[1]:
                    self.node.qmemory.operate(ns.X, 0)
                self.send_signal(Signals.SUCCESS, 0)
                fidelity = ns.qubits.fidelity(self.node.qmemory.peek(0)[0],
                                              ns.y0, squared=True)
                print(f"{ns.sim_time():.1f}: Bob received entangled qubit and "
                      f"corrections! Fidelity = {fidelity:.3f}")
                break

from netsquid.examples.teleportation import example_network_setup
ns.sim_reset()
ns.set_qstate_formalism(ns.QFormalism.DM)
ns.set_random_state(seed=42)
network = example_network_setup()
alice = network.get_node("Alice")
bob = network.get_node("Bob")
random_state_protocol = InitStateProtocol(alice)
bell_measure_protocol = BellMeasurementProtocol(alice, random_state_protocol)
correction_protocol = CorrectionProtocol(bob)
bell_measure_protocol.start()
correction_protocol.start()
stats = ns.sim_run(100)