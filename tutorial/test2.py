import netsquid as ns
import pydynaa
ns.set_random_state(seed=42)

#A quantum ping pong example
#Scheduling events
class PingEntity(pydynaa.Entity):
    ping_evtype = pydynaa.EventType("PING_EVENT", "A ping event.")
    delay = 10.

    def start(self, qubit): #ゲームを開始するイベント
        # Start the game by scheduling the first ping event after delay
        self.qubit = qubit #送受信する量子ビットを保存
        self._schedule_after(self.delay, PingEntity.ping_evtype) #イベントをスケジュールする _schdule_now()だと、即座にスケジュールできる

    def wait_for_pong(self, pong_entity):
        # Setup this entity to listen for pong events from a PongEntity
        pong_handler = pydynaa.EventHandler(self._handle_pong_event) #_handle_pong_eventをコールバック関数とする
        self._wait(pong_handler, entity=pong_entity,
                   event_type=PongEntity.pong_evtype) #pongがスケジュールしたイベントが来るまで待機

    def _handle_pong_event(self, event):
        # Callback function called by the pong handler when pong event is triggered
        m, prob = ns.qubits.measure(self.qubit, observable=ns.Z) #pongから来たqubitを測定
        labels_z = ("|0>", "|1>")
        print(f"{ns.sim_time():.1f}: Pong event! PingEntity measured "
              f"{labels_z[m]} with probability {prob:.2f}")
        self._schedule_after(PingEntity.delay, PingEntity.ping_evtype) #指定時間後にイベントをスケジュールする

class PongEntity(pydynaa.Entity):
    pong_evtype = pydynaa.EventType("PONG_EVENT", "A pong event.")
    delay = 10.

    def wait_for_ping(self, ping_entity):
        # Setup this entity to listen for ping events from a PingEntity
        ping_handler = pydynaa.EventHandler(self._handle_ping_event) #_handle_ping_eventをコールバック関数とする
        self._wait(ping_handler, entity=ping_entity,
                   event_type=PingEntity.ping_evtype) #pingがスケジュールしたイベントが来るまで待機

    def _handle_ping_event(self, event): #wait_for_pingが終わったら実行される
        # Callback function called by the ping handler when ping event is triggered
        m, prob = ns.qubits.measure(event.source.qubit, observable=ns.X) #pingから来たqubitを測定
        labels_x = ("|+>", "|->")
        print(f"{ns.sim_time():.1f}: Ping event! PongEntity measured "
              f"{labels_x[m]} with probability {prob:.2f}")
        self._schedule_after(PongEntity.delay, PongEntity.pong_evtype) #イベントをスケジュールする

# Create entities and register them to each other
ping = PingEntity()
pong = PongEntity()
ping.wait_for_pong(pong)
pong.wait_for_ping(ping)
#
# Create a qubit and instruct the ping entity to start
qubit, = ns.qubits.create_qubits(1)
ping.start(qubit)

stats = ns.sim_run(end_time=91) #シミュレーションを91ns間、実行する
print(stats)

#Event expressions by example: quantum teleportation*
ns.sim_reset()

class Charlie(pydynaa.Entity):
    ready_evtype = pydynaa.EventType("QUBITS_READY", "Entangled qubits are ready.")
    _generate_evtype = pydynaa.EventType("GENERATE", "Generate entangled qubits.")
    period = 50.
    delay = 10.

    def __init__(self):
        # Initialise Charlie by entangling qubits after every generation event
        self.entangled_qubits = None
        self._generate_handler = pydynaa.EventHandler(self._entangle_qubits) #callback = _entangle_qubits
        self._wait(self._generate_handler, entity=self,
                   event_type=Charlie._generate_evtype) #C_gイベントが来るまで待機

    def _entangle_qubits(self, event):
        # Callback function that entangles qubits and schedules an
        # entanglement ready event
        q1, q2 = ns.qubits.create_qubits(2)
        ns.qubits.operate(q1, ns.H)
        ns.qubits.operate([q1, q2], ns.CNOT)
        self.entangled_qubits = [q1, q2] #q1, q2はエンタングル状態
        self._schedule_after(Charlie.delay, Charlie.ready_evtype) #C_rイベントをスケジュールする
        print(f"{ns.sim_time():.1f}: Charlie finished generating entanglement")
        self._schedule_after(Charlie.period, Charlie._generate_evtype) #C_gイベントをスケジュールする

    def start(self):
        # Begin generating entanglement
        print(f"{ns.sim_time():.1f}: Charlie start generating entanglement")
        self._schedule_now(Charlie._generate_evtype) #C_gイベントをスケジュールする

class Alice(pydynaa.Entity):
    ready_evtype = pydynaa.EventType("CORRECTION_READY", "Corrections are ready.")
    _teleport_evtype = pydynaa.EventType("TELEPORT", "Teleport the qubit.")
    delay = 20.

    def __init__(self, teleport_state):
        # Initialise Alice by setting the teleport state and waiting to teleport
        self.teleport_state = teleport_state
        self.q0 = None
        self.q1 = None
        self.corrections = None
        self._teleport_handler = pydynaa.EventHandler(self._handle_teleport) #callback = _handle_teleport
        self._wait(self._teleport_handler, entity=self,
                   event_type=Alice._teleport_evtype) #A_tイベントが来るまで待機

    def wait_for_charlie(self, charlie):
        # Setup Alice to wait for an entanglement qubit from Charlie
        self._qubit_handler = pydynaa.EventHandler(self._handle_qubit) #callback = _handle_qubit
        self._wait(self._qubit_handler, entity=charlie,
                   event_type=Charlie.ready_evtype) #C_rイベントが来るまで待機

    def _handle_qubit(self, event):
        # Callback function that handles arrival of entangled qubit
        # and schedules teleportation
        self.q0, = ns.qubits.create_qubits(1, no_state=True) #伝えたい量子ビット
        self.q1 = event.source.entangled_qubits[0] #エンタングルした量子ビット
        ns.qubits.assign_qstate([self.q0], self.teleport_state)
        self._schedule_after(Alice.delay, Alice._teleport_evtype) #A_tイベントをスケジュールする
        print(f"{ns.sim_time():.1f}: Alice received entangled qubit")

    def _handle_teleport(self, event):
        # Callback function that does teleportation and schedules
        # a corrections ready event
        ns.qubits.operate([self.q0, self.q1], ns.CNOT)
        ns.qubits.operate(self.q0, ns.H)
        m0, __ = ns.qubits.measure(self.q0) #q0, q1を測定
        m1, __ = ns.qubits.measure(self.q1)
        self.corrections = [m0, m1]
        self._schedule_now(Alice.ready_evtype) #A_rイベントをスケジュールする
        print(f"{ns.sim_time():.1f}: Alice measured qubits & sending corrections")

class Bob(pydynaa.Entity):

    def wait_for_teleport(self, alice, charlie):
        # Setup Bob to wait for his entangled qubit and Alice's corrections
        charlie_ready_evexpr = pydynaa.EventExpression(
            source=charlie, event_type=Charlie.ready_evtype)
        alice_ready_evexpr = pydynaa.EventExpression(
            source=alice, event_type=Alice.ready_evtype)
        both_ready_evexpr = charlie_ready_evexpr & alice_ready_evexpr
        self._teleport_handler = pydynaa.ExpressionHandler(self._handle_teleport) #callback = _handle_teleport
        self._wait(self._teleport_handler, expression=both_ready_evexpr) #A_rとC_rイベントが来るまで待機

    def _handle_teleport(self, event_expression):
        # Callback function that handles messages from both Alice and Charlie
        qubit = event_expression.first_term.atomic_source.entangled_qubits[1]
        alice = event_expression.second_term.atomic_source
        self._apply_corrections(qubit, alice.corrections)

    def _apply_corrections(self, qubit, corrections):
        # Apply teleportation corrections and check fidelity
        m0, m1 = corrections
        if m1:
            ns.qubits.operate(qubit, ns.X)
        if m0:
            ns.qubits.operate(qubit, ns.Z)
        fidelity = ns.qubits.fidelity(qubit, alice.teleport_state, squared=True) #元の量子ビットと同じか比較
        print(f"{ns.sim_time():.1f}: Bob received entangled qubit and corrections!"
              f" Fidelity = {fidelity:.3f}")

def setup_network(alice, bob, charlie):
    alice.wait_for_charlie(charlie)
    bob.wait_for_teleport(alice, charlie)
    charlie.start()

alice = Alice(teleport_state=ns.h1)
bob = Bob()
charlie = Charlie()

setup_network(alice, bob, charlie)
stats = ns.sim_run(end_time=100)

print(stats)

class NoisyBob(Bob):
    depolar_rate = 1e7  # depolarization rate of waiting qubits [Hz]

    def _handle_teleport(self, event_expression):
        # Callback function that first applies noise to qubit before corrections
        alice_expr = event_expression.second_term
        charlie_expr = event_expression.first_term
        # Compute time that qubit from Charlie has been waiting:
        delay = ns.sim_time() - charlie_expr.triggered_time
        # Apply time-dependent quantum noise to Bob's qubit:
        qubit = charlie_expr.atomic_source.entangled_qubits[1]
        ns.qubits.delay_depolarize(qubit, NoisyBob.depolar_rate, delay)
        # Apply classical corrections (as before):
        self._apply_corrections(qubit, alice_expr.atomic_source.corrections)

ns.sim_reset()
ns.set_qstate_formalism(ns.QFormalism.DM)

alice = Alice(teleport_state=ns.h1)
bob = NoisyBob()
charlie = Charlie()
setup_network(alice, bob, charlie)
stats = ns.sim_run(end_time=50)