import netsquid as ns
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.examples.repeater import example_network_setup, example_sim_setup

print("This example module is located at: "
      "{}".format(ns.examples.repeater.__file__))

network = example_network_setup()
repeater_example, dc = example_sim_setup(
    network.get_node("node_A"), network.get_node("node_B"),
    network.get_node("node_R"), num_runs=1000)
repeater_example.start()
ns.sim_run()
print("Average fidelity of generated entanglement via a repeater "
      "and with filtering: {}".format(dc.dataframe["F2"].mean()))