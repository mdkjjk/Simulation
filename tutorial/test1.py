import netsquid as ns

#Qubits and their quantum state
q1, q2 = ns.qubits.create_qubits(2) #qubitを生成

q1.qstate.num_qubits #q1の状態に含まれる量子ビット数を示す
q1.qstate.qrepr #qubitの状態を表示
q2.qstate.qrepr

print(q1.qstate == q2.qstate) #もつれ状態にあるか確認
ns.qubits.combine_qubits([q1,q2]) #もつれ状態にする（テンソル積を求める） 
print(q1.qstate == q2.qstate)
q1.qstate.num_qubits

ns.qubits.reduced_dm([q1, q2]) #縮小密度行列を求める なんで4＊4の行列になるのか？
print(ns.qubits.reduced_dm(q2))

#Qubit measurement
q1.qstate.num_qubits == q2.qstate.num_qubits #q1 & q2がエンタングルしている
print(q1.qstate.num_qubits)
print(ns.qubits.measure(q1)) #qubitを測定=>測定結果 & その結果になる確率をreturn
print(q1.qstate.num_qubits) #2->1になるのはなんで？ =>測定したことでもつれ状態でなくなったから
print(ns.qubits.reduced_dm(q1))

ns.qubits.combine_qubits([q1, q2])
print(q1.qstate.num_qubits)
print(ns.qubits.measure(q2, discard=True)) #測定後、qubitを破棄
print(q2.qstate is None)
print(q1.qstate.num_qubits)

#Quantum state formalism
from netsquid.qubits.qformalism import QFormalism
print(ns.get_qstate_formalism()) #qubitの状態形式を表示
# Change to stabilizer formalism:
ns.set_qstate_formalism(QFormalism.STAB) #qubitの状態形式を定義　デフォはKet vector
print(ns.get_qstate_formalism())

q1, q2 = ns.qubits.create_qubits(2, no_state=True) #状態なしのqubitを生成
ns.qubits.assign_qstate([q1, q2], ns.h01)  # assign |+->　割り当てる
print(ns.qubits.reduced_dm(q1))
print(ns.qubits.reduced_dm(q2))

print(type(q1.qstate.qrepr))
print(q1.qstate.qrepr.check_matrix)
print(q1.qstate.qrepr.phases)

#Quantum operations
# Change to density matrix formalism:
ns.set_qstate_formalism(QFormalism.DM)
a1, a2, b1 = ns.qubits.create_qubits(3)

# put a1 into the chosen target state
ns.qubits.operate(a1, ns.H) # Hゲートをa1に適用: |0> -> |+>
ns.qubits.operate(a1, ns.S) # Sゲートをa1に適用: |+> -> |0_y>
print(ns.qubits.reduced_dm([a1]))
#a2とb1をベル状態 |b00> = (|00> + |11>)/sqrt(2) に変換
ns.qubits.operate(a2, ns.H)  # Hゲートをa2に適用
ns.qubits.operate([a2, b1], ns.CNOT)  # CNOT: a2 = 制御ビット, b1 = ターゲットビット
print(ns.qubits.reduced_dm([a2, b1]))

import numpy as np
# Construct a new operator using existing operators:
newOp = ns.CNOT * ((ns.X + ns.Z) / np.sqrt(2) ^ ns.I)
print(newOp.name)  # Note: CNOT == CX 数式の表示
# Construct a new operator using a matrix:
newOp2 = ns.qubits.Operator("newOp2", np.array([[1, 1j], [-1j, -1]])/np.sqrt(2))
assert(newOp2.is_unitary == True)
assert(newOp2.is_hermitian == True)
# Construct new operators using helper functions:
R = ns.create_rotation_op(angle=np.pi/4, rotation_axis=(1, 0, 0))
print(R.name)
# Construct a controlled operator:
CR = R.ctrl
print(CR.name)

ns.set_random_state(seed=42)  # (Ensures fixed random outcomes for our doctests)
ns.qubits.operate([a1, a2], ns.CNOT)  # CNOT: a1 = 制御ビット, a2 = ターゲットビット
ns.qubits.operate(a1, ns.H)
# Measure a1 in the standard basis:
m1, prob = ns.qubits.measure(a1)
labels_z = ("|0>", "|1>")
print(f"Measured {labels_z[m1]} with prob {prob:.2f}")
# Measure a2 in standard basis:
m2, prob = ns.qubits.measure(a2)
print(f"Measured {labels_z[m2]} with prob {prob:.2f}")

q1, = ns.qubits.create_qubits(1)
m3, prob = ns.qubits.measure(q1, observable=ns.X, discard=True)
labels_x = ("+", "-")
print(f"Measured |{labels_x[m3]}> with prob {prob:.2f}")

#標準基底とハダマート基底を交互で測定可能
q1, = ns.qubits.create_qubits(1)
for i in range(6):
    observable, labels = (ns.Z, ("0", "1")) if i % 2 else (ns.X, ("+", "-"))
    m, prob = ns.qubits.measure(q1, observable=observable)
    print(f"Measured |{labels[m]}> with prob {prob:.2f}")

#(General measurements*)
bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
q1, q2 = ns.qubits.create_qubits(2)
ns.qubits.operate(q1, ns.H)
meas, prob = ns.qubits.gmeasure([q1, q2], meas_operators=bell_operators)
labels_bell = ("|00>", "|01>", "|10>", "|11>")
print(f"Measured {labels_bell[meas]} with prob {prob:.2f}")
print(q1.qstate.num_qubits)

if m2 == 1:
    ns.qubits.operate(b1, ns.X)
if m1 == 1:
    ns.qubits.operate(b1, ns.Z)
print(ns.qubits.reduced_dm([b1]))

fidelity = ns.qubits.fidelity(b1, ns.y0, squared=True)
print(f"Fidelity is {fidelity:.3f}") #元のqubitとの適合度を表示


#(Applying noise*)
ns.qubits.delay_depolarize(b1, depolar_rate=1e7, delay=20) #遅延を考慮することも可能
fidelity = ns.qubits.fidelity([b1], reference_state=ns.y0, squared=True)
print(f"Fidelity is {fidelity:.3f}")

q1, q2, q3, q4 = ns.qubits.create_qubits(4)
ns.qubits.stochastic_operate(q1, [ns.X, ns.Y, ns.Z], p_weights=(1/2, 1/4, 1/4))
print(ns.qubits.reduced_dm([q1]))
ns.qubits.apply_pauli_noise(q2, p_weights=(1/4, 1/4, 1/4, 1/4))  # (I, X, Y, Z)
print(ns.qubits.reduced_dm([q2]))
ns.qubits.depolarize(q3, prob=0.8)
print(ns.qubits.reduced_dm([q3]))
ns.qubits.operate(q4, ns.X)  # -> |1>
ns.qubits.amplitude_dampen(q4, gamma=0.1, prob=1)
print(ns.qubits.reduced_dm([q4]))