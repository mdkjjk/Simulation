import netsquid as ns
from netsquid.examples.purify import example_network_setup, example_sim_setup

print("This example module is located at: {}".format(ns.examples.purify.__file__))

network = example_network_setup()
filt_example, dc = example_sim_setup(
    network.get_node("node_A"), network.get_node("node_B"), num_runs=1000)
filt_example.start()
ns.sim_run()
print("Average fidelity of generated entanglement with filtering: {}"
      .format(dc.dataframe["F2"].mean()))