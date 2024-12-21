import netsquid as ns
from netsquid.examples.entanglenodes import example_network_setup, EntangleNodes

print("This example module is located at: "
      "{}".format(ns.examples.entanglenodes.__file__))

network = example_network_setup()
protocol_a = EntangleNodes(node=network.subcomponents["node_A"], role="source")
protocol_b = EntangleNodes(node=network.subcomponents["node_B"], role="receiver")
protocol_a.start()
protocol_b.start()
ns.sim_run()
q1, = network.subcomponents["node_A"].qmemory.peek(0)
q2, = network.subcomponents["node_B"].qmemory.peek(0)
print("Fidelity of generated entanglement: {}".format(
      ns.qubits.fidelity([q1, q2], ns.b00)))