import netsquid as ns
from netsquid.qubits.qformalism import QFormalism

ns.set_qstate_formalism(QFormalism.DM)

q1, q2 = ns.qubits.create_qubits(2, no_state=True) #状態なしのqubitを生成
ns.qubits.assign_qstate([q1, q2], ns.b01)

#print(f"{q1.qstate.qrepr}")
#print(f"{q2.qstate.qrepr}")

ns.qubits.delay_depolarize(q1, depolar_rate=1000, delay=1e5)
print(f"{q1.qstate.qrepr}")
print(f"{q2.qstate.qrepr}")
print(f"Fidelity = {ns.qubits.fidelity([q1, q2], ns.b01, squared=True)}")