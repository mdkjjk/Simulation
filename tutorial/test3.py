import netsquid as ns
from netsquid.components import Channel, QuantumChannel
channel = Channel(name="MyChannel")

channel.send("hello world!") #send()でメッセージを送る
ns.sim_run()

mess, delay = channel.receive() #receive()は出力されたメッセージと、メッセージがチャネルを通過した時間を返す
print(mess)
print(delay) #デフォルトで遅延はなし

#遅延を追加する方法１
chan = Channel(name="DelayChannel", delay=10) #チャネル作成の時に固定値の遅延を設定する

chan.send("How are you?")
ns.sim_run()

mess, delay = chan.receive()
print(mess)
print(delay)


#遅延を追加する方法２
from netsquid.components.models.delaymodels import FixedDelayModel
fixed_delaymodel = FixedDelayModel(delay=10) #delay modelを作成する 固定値の遅延を設定

channel.models['delay_model'] = fixed_delaymodel
channel.send("hello world!")
ns.sim_run()
print(channel.receive())

from netsquid.components.models.delaymodels import GaussianDelayModel #Gaussian delay model
gaussian_delaymodel = GaussianDelayModel(delay_mean=5, delay_std=0.1) #ランダムな遅延を設定

channel.models['delay_model'] = gaussian_delaymodel
channel.send("How are you?")
ns.sim_run()
print(channel.receive())

Channel("TutorialChannel", length=10) #初期に長さの設定可能
channel.properties['length'] = 5 #lengthは後から変更可能
print(channel.properties['length'])

from netsquid.components.models.delaymodels import FibreDelayModel #Fibre delay model
delay_model = FibreDelayModel() #FibreDelayModel は、光ファイバーケーブルに存在する遅延をモデル化したもの
print(f"Speed of light in fibre: {delay_model.properties['c']:.1f} [km/s]") #ケーブル内の光速と距離に依存して遅延が変化
print(delay_model.required_properties) #必要となる要素を表示

channel.models['delay_model'] = delay_model
channel.send("hello world!")
ns.sim_run()
print(channel.receive())

#量子チャネルにのみ考慮されるモデル => quantum_noise & quantum_loss
#quantum lossを設定
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.qchannel import QuantumChannel
loss_model = FibreLossModel(p_loss_init=0.83, p_loss_length=0.2)
qchannel = QuantumChannel("MyQChannel", length=20, models={'quantum_loss_model': loss_model})

#Quantum memory
from netsquid.components import QuantumMemory
qmem = QuantumMemory(name="MyMemory", num_positions=1) #num_positionsは量子ビットを保存できるメモリの数

from netsquid.components.models.qerrormodels import DepolarNoiseModel #脱分極ノイズを加えるモデル
depolar_noise = DepolarNoiseModel(depolar_rate=1e6)  # the depolar_rate is in Hz

qmem = QuantumMemory("DepolarMemory", num_positions=2,
    memory_noise_models=[depolar_noise, depolar_noise]) #メモリごとに異なるエラーモデルを割り当てられる

for mem_pos in qmem.mem_positions:
    mem_pos.models['noise_model'] = depolar_noise #for文で割り当てることもできる

from netsquid.qubits.qubitapi import create_qubits
qubits = create_qubits(1)
qmem.put(qubits) #putでメモリに量子ビットを挿入
print(qmem.peek(0))
print(qmem.pop(positions=0)) #popでメモリから量子ビットを取り出す
print(qmem.peek(0)) #peekでメモリに量子ビットがあるか確認できる

#メモリに操作を加えたり、測定したりできる
import netsquid.qubits.operators as ops
qmem.put(qubits)
qmem.operate(ops.X, positions=[0]) #メモリポジション０に保存されている量子ビットにXゲートを適用

print(qmem.measure(positions=[0])) #メモリポジション０に保存されている量子ビットを測定
print(qmem.measure(positions=[0], observable=ops.X) )

#Ports
channel = Channel("TutorialChannel", delay=3)
channel.ports['send'].tx_input("hello") #入力を送信
ns.sim_run()
print(channel.ports['recv'].rx_output()) #出力を受信

channel.ports['recv'].connect(qmem.ports['qin0']) #connect()で２つのポートを接続できる

qubit, = create_qubits(1)
print(qubit)
channel.send(qubit) #ポートが接続されているときは、send()するだけで自動的に量子メモリに保存される
ns.sim_run()
print(qmem.peek(0))

#Ping pong using components and ports
#コンポーネントとポートを利用したPing-Pongゲーム
from netsquid.components.component import Port
import pydynaa

class PingEntity(pydynaa.Entity):
    length = 2e-3  # channel length [km]

    def __init__(self):
        # Create a memory and a quantum channel:
        self.qmemory = QuantumMemory("PingMemory", num_positions=1)
        self.qchannel = QuantumChannel("PingChannel", length=self.length,
                                       models={"delay_model": FibreDelayModel()})
        # link output from qmemory (pop) to input of ping channel:
        self.qmemory.ports["qout"].connect(self.qchannel.ports["send"])
        # Setup callback function to handle input on quantum memory port "qin0":
        self._wait(pydynaa.EventHandler(self._handle_input_qubit),
                   entity=self.qmemory.ports["qin0"], event_type=Port.evtype_input)
        self.qmemory.ports["qin0"].notify_all_input = True

    def start(self, qubit):
        # Start the game by having ping player send the first qubit (ping)
        self.qchannel.send(qubit)

    def wait_for_pong(self, other_entity):
        # Setup this entity to pass incoming qubits to its quantum memory
        self.qmemory.ports["qin0"].connect(other_entity.qchannel.ports["recv"])

    def _handle_input_qubit(self, event):
        # Callback function called by the pong handler when pong event is triggered
        [m], [prob] = self.qmemory.measure(positions=[0], observable=ns.Z)
        labels_z = ("|0>", "|1>")
        print(f"{ns.sim_time():.1f}: Pong event! PingEntity measured "
              f"{labels_z[m]} with probability {prob:.2f}")
        self.qmemory.pop(positions=[0])

class PongEntity(pydynaa.Entity):
    length = 2e-3  # channel length [km]

    def __init__(self):
        # Create a memory and a quantum channel:
        self.qmemory = QuantumMemory("PongMemory", num_positions=1)
        self.qchannel = QuantumChannel("PingChannel", length=self.length,
                                       models={"delay_model": FibreDelayModel()})
        # link output from qmemory (pop) to input of ping channel:
        self.qmemory.ports["qout"].connect(self.qchannel.ports["send"])
        # Setup callback function to handle input on quantum memory:
        self._wait(pydynaa.EventHandler(self._handle_input_qubit),
                   entity=self.qmemory.ports["qin0"], event_type=Port.evtype_input)
        self.qmemory.ports["qin0"].notify_all_input = True

    def wait_for_ping(self, other_entity):
        # Setup this entity to pass incoming qubits to its quantum memory
        self.qmemory.ports["qin0"].connect(other_entity.qchannel.ports["recv"])

    def _handle_input_qubit(self, event):
        # Callback function called by the pong handler when pong event is triggered
        [m], [prob] = self.qmemory.measure(positions=[0], observable=ns.X)
        labels_x = ("|+>", "|->")
        print(f"{ns.sim_time():.1f}: Ping event! PongEntity measured "
              f"{labels_x[m]} with probability {prob:.2f}")
        self.qmemory.pop(positions=[0])

# Create entities and register them to each other
ns.sim_reset()
ping = PingEntity()
pong = PongEntity()
ping.wait_for_pong(pong)
pong.wait_for_ping(ping)
# Create a qubit and instruct the ping entity to start
qubit, = ns.qubits.create_qubits(1)
ping.start(qubit)

ns.set_random_state(seed=42)
stats = ns.sim_run(91)

#Quantum telportation using components
ns.set_qstate_formalism(ns.QFormalism.DM)
ns.sim_reset()

class Alice(pydynaa.Entity):
    def __init__(self, teleport_state, cchannel_send_port):
        self.teleport_state = teleport_state
        self.cchannel_send_port = cchannel_send_port
        self.qmemory = QuantumMemory("AliceMemory", num_positions=2)
        self._wait(pydynaa.EventHandler(self._handle_input_qubit),
                   entity=self.qmemory.ports["qin1"], event_type=Port.evtype_input) #ポートqin1に入ってくるまで待機
        self.qmemory.ports["qin1"].notify_all_input = True #このポートが接続、転送、バインドされたとき、常に入力イベントをスケジュールする

    def _handle_input_qubit(self, event):
        # Callback function that does teleportation and
        # schedules a corrections ready event
        q0, = ns.qubits.create_qubits(1, no_state=True)
        ns.qubits.assign_qstate([q0], self.teleport_state)
        self.qmemory.put([q0], positions=[0])
        self.qmemory.operate(ns.CNOT, positions=[0, 1])
        self.qmemory.operate(ns.H, positions=[0])
        m0, m1 = self.qmemory.measure(positions=[0, 1], observable=ns.Z,
                                      discard=True)[0]
        self.cchannel_send_port.tx_input([m0, m1])
        print(f"{ns.sim_time():.1f}: Alice received entangled qubit, "
              f"measured qubits & sending corrections")

class Bob(pydynaa.Entity):
    depolar_rate = 1e7  # depolarization rate of waiting qubits [Hz]

    def __init__(self, cchannel_recv_port):
        noise_model = DepolarNoiseModel(depolar_rate=self.depolar_rate)
        self.qmemory = QuantumMemory("BobMemory", num_positions=1,
                                     memory_noise_models=[noise_model])
        cchannel_recv_port.bind_output_handler(self._handle_corrections) #recvポートがメッセージを受け取ったら、_handle_correctionsが実行

    def _handle_corrections(self, message):
        # Callback function that handles messages from both Alice and Charlie
        m0, m1 = message.items
        if m1:
            self.qmemory.operate(ns.X, positions=[0])
        if m0:
            self.qmemory.operate(ns.Z, positions=[0])
        qubit = self.qmemory.pop(positions=[0])
        fidelity = ns.qubits.fidelity(qubit, ns.y0, squared=True)
        print(f"{ns.sim_time():.1f}: Bob received entangled qubit and corrections!"
              f" Fidelity = {fidelity:.3f}")

from netsquid.qubits.state_sampler import StateSampler
import netsquid.qubits.ketstates as ks
state_sampler = StateSampler([ks.b00], [1.0]) #エンタングル状態の量子ビットを生成

from netsquid.components.qsource import QSource, SourceStatus
charlie_source = QSource("Charlie", state_sampler, frequency=100, num_ports=2,
                         timing_model=FixedDelayModel(delay=50),
                         status=SourceStatus.INTERNAL) #ポートが２つあるのは、生成された量子ビットをAliceとBobに送りたいから

def setup_network(alice, bob, qsource, length=4e-3):
    qchannel_c2a = QuantumChannel("Charlie->Alice", length=length / 2,
                                  models={"delay_model": FibreDelayModel()})
    qchannel_c2b = QuantumChannel("Charlie->Bob", length=length / 2,
                                  models={"delay_model": FibreDelayModel()})
    qsource.ports['qout0'].connect(qchannel_c2a.ports['send'])
    qsource.ports['qout1'].connect(qchannel_c2b.ports['send'])
    alice.qmemory.ports['qin1'].connect(qchannel_c2a.ports['recv'])
    bob.qmemory.ports['qin0'].connect(qchannel_c2b.ports['recv'])

from netsquid.components import ClassicalChannel
cchannel = ClassicalChannel("CChannel", length=4e-3,
                            models={"delay_model": FibreDelayModel()})
alice = Alice(teleport_state=ns.y0, cchannel_send_port=cchannel.ports["send"])
bob = Bob(cchannel_recv_port=cchannel.ports["recv"])
setup_network(alice, bob, charlie_source)

stats = ns.sim_run(end_time=100)