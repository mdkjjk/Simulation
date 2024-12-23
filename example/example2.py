import netsquid as ns
from netsquid.examples.purify import example_network_setup, Distil

def run_distillation_simulation(num_runs=1000):
    # ネットワークをセットアップ
    network = example_network_setup()
    node_a = network.get_node("node_A")
    node_b = network.get_node("node_B")

    # ノード間の古典通信ポートを取得
    port_a = node_a.get_conn_port(node_b.ID)
    port_b = node_b.get_conn_port(node_a.ID)

    # DistilプロトコルをノードAとノードBに設定
    distil_a = Distil(node=node_a, port=port_a, role="A", name="Distil_A")
    distil_b = Distil(node=node_b, port=port_b, role="B", name="Distil_B")

    # プロトコルを開始
    distil_a.start()
    distil_b.start()

    # シミュレーションを実行
    for run in range(num_runs):
        ns.sim_run()

        # 成功したエンタングルメントの位置を取得
        if distil_a.is_connected and distil_b.is_connected:
            print(f"Run {run + 1}: Distillation succeeded!")
        else:
            print(f"Run {run + 1}: Distillation failed.")

    # プロトコルを停止
    distil_a.stop()
    distil_b.stop()

if __name__ == "__main__":
    run_distillation_simulation()
