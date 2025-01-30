from qiskit.circuit.classicalfunction import classical_function
from qiskit.circuit.classicalfunction.types import Int1

@classical_function
def cf_v2_c1(x0 : Int1, x1 : Int1) -> Int1:
    return (x1) ^ (x0)
    
def pf_v2_c1(x0, x1):
    return (x1) ^ (x0)
    

@classical_function
def cf_v2_c2(x0 : Int1, x1 : Int1) -> Int1:
    return ((x1) and (x0)) and (x0)
    
def pf_v2_c2(x0, x1):
    return ((x1) and (x0)) and (x0)
    

@classical_function
def cf_v2_c3(x0 : Int1, x1 : Int1) -> Int1:
    return not (((x1) ^ (x0)) or ((x0) ^ (x0)))
    
def pf_v2_c3(x0, x1):
    return not (((x1) ^ (x0)) or ((x0) ^ (x0)))
    

@classical_function
def cf_v2_c4(x0 : Int1, x1 : Int1) -> Int1:
    return ((not (not (x1))) or (not (x0))) ^ ((not (x1)) or (not (x1)))
    
def pf_v2_c4(x0, x1):
    return ((not (not (x1))) or (not (x0))) ^ ((not (x1)) or (not (x1)))
    

@classical_function
def cf_v2_c5(x0 : Int1, x1 : Int1) -> Int1:
    return ((not (((x1) and (x1)) and (x1))) and ((x0) and (x1))) ^ (not (((x0) and (x1)) and ((x0) and (x1))))
    
def pf_v2_c5(x0, x1):
    return ((not (((x1) and (x1)) and (x1))) and ((x0) and (x1))) ^ (not (((x0) and (x1)) and ((x0) and (x1))))
    

@classical_function
def cf_v2_c6(x0 : Int1, x1 : Int1) -> Int1:
    return ((((((x0) and (x1)) or (x1)) ^ (((x1) and (x0)) or ((x1) and (x0)))) ^ (x0)) or (((x1) and (x0)) ^ (((x1) and (x0)) or ((x1) and (x0))))) or ((x1) and (x0))
    
def pf_v2_c6(x0, x1):
    return ((((((x0) and (x1)) or (x1)) ^ (((x1) and (x0)) or ((x1) and (x0)))) ^ (x0)) or (((x1) and (x0)) ^ (((x1) and (x0)) or ((x1) and (x0))))) or ((x1) and (x0))
    

@classical_function
def cf_v2_c7(x0 : Int1, x1 : Int1) -> Int1:
    return not ((((((not (x1)) or (not (x1))) and ((x0) or (not (x1)))) and (x0)) ^ (x0)) and (((x1) and ((x0) or (not (x1)))) ^ (not (x1))))
    
def pf_v2_c7(x0, x1):
    return not ((((((not (x1)) or (not (x1))) and ((x0) or (not (x1)))) and (x0)) ^ (x0)) and (((x1) and ((x0) or (not (x1)))) ^ (not (x1))))
    

@classical_function
def cf_v2_c8(x0 : Int1, x1 : Int1) -> Int1:
    return ((not (not (((not ((x0) ^ (x1))) ^ (not ((x1) ^ (x1)))) and ((x1) ^ (x1))))) ^ (x1)) and (not ((x1) ^ (x1)))
    
def pf_v2_c8(x0, x1):
    return ((not (not (((not ((x0) ^ (x1))) ^ (not ((x1) ^ (x1)))) and ((x1) ^ (x1))))) ^ (x1)) and (not ((x1) ^ (x1)))
    

@classical_function
def cf_v2_c9(x0 : Int1, x1 : Int1) -> Int1:
    return not (not (not (((((((x1) or (x1)) ^ ((x0) or (x0))) ^ (x1)) ^ (x1)) and ((x1) ^ ((x0) or (x0)))) and ((x0) and (x1)))))
    
def pf_v2_c9(x0, x1):
    return not (not (not (((((((x1) or (x1)) ^ ((x0) or (x0))) ^ (x1)) ^ (x1)) and ((x1) ^ ((x0) or (x0)))) and ((x0) and (x1)))))
    

@classical_function
def cf_v2_c10(x0 : Int1, x1 : Int1) -> Int1:
    return (((not ((((not (not ((x1) and (x1)))) or (x0)) ^ ((x0) or (x0))) and (x0))) and ((x0) or (x0))) or (x0)) and ((not ((x0) and (x1))) and (not (x1)))
    
def pf_v2_c10(x0, x1):
    return (((not ((((not (not ((x1) and (x1)))) or (x0)) ^ ((x0) or (x0))) and (x0))) and ((x0) or (x0))) or (x0)) and ((not ((x0) and (x1))) and (not (x1)))
    

@classical_function
def cf_v2_c11(x0 : Int1, x1 : Int1) -> Int1:
    return not (not ((((((((((x1) or (x0)) or ((x1) or (x1))) and (((x1) or (x1)) or (x0))) or ((x1) or (x1))) or ((x1) or (x1))) ^ ((x1) or (x0))) ^ ((((x1) or (x1)) and (((x1) or (x1)) or (x0))) or ((x1) or (x0)))) and ((x1) or (x0))) or (((x1) or (x1)) and (((x1) or (x1)) or (x0)))))
    
def pf_v2_c11(x0, x1):
    return not (not ((((((((((x1) or (x0)) or ((x1) or (x1))) and (((x1) or (x1)) or (x0))) or ((x1) or (x1))) or ((x1) or (x1))) ^ ((x1) or (x0))) ^ ((((x1) or (x1)) and (((x1) or (x1)) or (x0))) or ((x1) or (x0)))) and ((x1) or (x0))) or (((x1) or (x1)) and (((x1) or (x1)) or (x0)))))
    

@classical_function
def cf_v2_c12(x0 : Int1, x1 : Int1) -> Int1:
    return (not (((((not (((((not (x1)) and (x1)) ^ (x1)) and (not (x0))) ^ (x0))) or ((x0) and (x1))) and (x1)) or ((x0) and (x0))) ^ ((not (not (x0))) and (not (not (x0)))))) ^ (x1)
    
def pf_v2_c12(x0, x1):
    return (not (((((not (((((not (x1)) and (x1)) ^ (x1)) and (not (x0))) ^ (x0))) or ((x0) and (x1))) and (x1)) or ((x0) and (x0))) ^ ((not (not (x0))) and (not (not (x0)))))) ^ (x1)
    

@classical_function
def cf_v2_c13(x0 : Int1, x1 : Int1) -> Int1:
    return (not ((((((not (((((not (x0)) and (not (x1))) and (x1)) or ((x0) and (x1))) or (not (x1)))) ^ ((x0) and (x1))) or ((not (x1)) and (x0))) and (not (x1))) and ((x0) or (x1))) or (not (x1)))) ^ (not (x1))
    
def pf_v2_c13(x0, x1):
    return (not ((((((not (((((not (x0)) and (not (x1))) and (x1)) or ((x0) and (x1))) or (not (x1)))) ^ ((x0) and (x1))) or ((not (x1)) and (x0))) and (not (x1))) and ((x0) or (x1))) or (not (x1)))) ^ (not (x1))
    

@classical_function
def cf_v2_c14(x0 : Int1, x1 : Int1) -> Int1:
    return ((not ((((((not ((((((x1) ^ (x1)) ^ (x0)) and ((x1) ^ (x1))) and ((x1) ^ (x1))) and (((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))))) or ((((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) or (x0))) ^ ((x1) and (((x1) ^ (x0)) or (x0)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or ((x1) and (((x1) ^ (x0)) or (x0)))) ^ (not (x0))
    
def pf_v2_c14(x0, x1):
    return ((not ((((((not ((((((x1) ^ (x1)) ^ (x0)) and ((x1) ^ (x1))) and ((x1) ^ (x1))) and (((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))))) or ((((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) or (x0))) ^ ((x1) and (((x1) ^ (x0)) or (x0)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or ((x1) and (((x1) ^ (x0)) or (x0)))) ^ (not (x0))
    

@classical_function
def cf_v2_c15(x0 : Int1, x1 : Int1) -> Int1:
    return not (((((not (not ((((((not (not (not (x0)))) ^ (not (x0))) and (not (x0))) and (x1)) or (x1)) and (x0)))) and ((x0) and (not (x0)))) ^ ((((not (x0)) ^ (not (x0))) or (x0)) and ((not (x0)) and (((not (x0)) ^ (not (x0))) or (x0))))) and (((not (x0)) ^ (not (x0))) or (x0))) ^ ((not (x0)) and (not ((x0) and (not (x0))))))
    
def pf_v2_c15(x0, x1):
    return not (((((not (not ((((((not (not (not (x0)))) ^ (not (x0))) and (not (x0))) and (x1)) or (x1)) and (x0)))) and ((x0) and (not (x0)))) ^ ((((not (x0)) ^ (not (x0))) or (x0)) and ((not (x0)) and (((not (x0)) ^ (not (x0))) or (x0))))) and (((not (x0)) ^ (not (x0))) or (x0))) ^ ((not (x0)) and (not ((x0) and (not (x0))))))
    

@classical_function
def cf_v2_c16(x0 : Int1, x1 : Int1) -> Int1:
    return not (not (((not ((not (not ((not (((((not ((x1) or (x0))) or ((x0) or (x0))) and ((x0) or (x0))) and (not ((x0) or (x0)))) and (((x0) or (x0)) and (not ((x0) or (x0)))))) and ((x1) and (not ((x0) or (x0))))))) and ((x0) or ((x0) or (x0))))) or (((x0) or ((x0) or (x0))) and ((x0) or (x0)))) or (((x0) or (x0)) and (not ((x0) or (x0))))))
    
def pf_v2_c16(x0, x1):
    return not (not (((not ((not (not ((not (((((not ((x1) or (x0))) or ((x0) or (x0))) and ((x0) or (x0))) and (not ((x0) or (x0)))) and (((x0) or (x0)) and (not ((x0) or (x0)))))) and ((x1) and (not ((x0) or (x0))))))) and ((x0) or ((x0) or (x0))))) or (((x0) or ((x0) or (x0))) and ((x0) or (x0)))) or (((x0) or (x0)) and (not ((x0) or (x0))))))
    

@classical_function
def cf_v3_c2(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((x0) ^ (x1)) and (x2)
    
def pf_v3_c2(x0, x1, x2):
    return ((x0) ^ (x1)) and (x2)
    

@classical_function
def cf_v3_c3(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return (not ((x1) and (x0))) or ((x1) and (x2))
    
def pf_v3_c3(x0, x1, x2):
    return (not ((x1) and (x0))) or ((x1) and (x2))
    

@classical_function
def cf_v3_c4(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not ((((x1) and (x1)) and (x2)) or (x0))
    
def pf_v3_c4(x0, x1, x2):
    return not ((((x1) and (x1)) and (x2)) or (x0))
    

@classical_function
def cf_v3_c5(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((((not (x0)) ^ (x0)) or ((x2) ^ (x0))) or (x2)) and (((x2) ^ (x0)) or (x1))
    
def pf_v3_c5(x0, x1, x2):
    return ((((not (x0)) ^ (x0)) or ((x2) ^ (x0))) or (x2)) and (((x2) ^ (x0)) or (x1))
    

@classical_function
def cf_v3_c6(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((((((x0) ^ (x1)) or (x2)) and (x2)) or (x2)) and (x1)) and ((x2) or ((x0) ^ (x1)))
    
def pf_v3_c6(x0, x1, x2):
    return ((((((x0) ^ (x1)) or (x2)) and (x2)) or (x2)) and (x1)) and ((x2) or ((x0) ^ (x1)))
    

@classical_function
def cf_v3_c7(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return (not (not (((((x2) or (x2)) ^ (x2)) or ((x2) ^ ((x2) or (x0)))) or ((x1) or (x1))))) and (not (x0))
    
def pf_v3_c7(x0, x1, x2):
    return (not (not (((((x2) or (x2)) ^ (x2)) or ((x2) ^ ((x2) or (x0)))) or ((x1) or (x1))))) and (not (x0))
    

@classical_function
def cf_v3_c8(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((((((not ((x0) ^ (x0))) or (x2)) ^ ((not (x1)) or (x0))) ^ ((x2) ^ (x1))) ^ ((x2) ^ ((x2) ^ (x1)))) or ((x2) ^ (x1))) and ((not (x1)) or (x2))
    
def pf_v3_c8(x0, x1, x2):
    return ((((((not ((x0) ^ (x0))) or (x2)) ^ ((not (x1)) or (x0))) ^ ((x2) ^ (x1))) ^ ((x2) ^ ((x2) ^ (x1)))) or ((x2) ^ (x1))) and ((not (x1)) or (x2))
    

@classical_function
def cf_v3_c9(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not (((not ((((not (not (x0))) or (not (not (x1)))) or (not (x1))) ^ (x0))) ^ (not ((x2) or (x2)))) ^ (not (not (x1))))
    
def pf_v3_c9(x0, x1, x2):
    return not (((not ((((not (not (x0))) or (not (not (x1)))) or (not (x1))) ^ (x0))) ^ (not ((x2) or (x2)))) ^ (not (not (x1))))
    

@classical_function
def cf_v3_c10(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not (((not (((((((x0) and (x1)) or ((x0) and (x2))) ^ (x0)) and (((x2) or (x0)) ^ (x0))) ^ ((x0) and (x2))) ^ ((x0) and (x2)))) and ((x2) or (x0))) or (x0))
    
def pf_v3_c10(x0, x1, x2):
    return not (((not (((((((x0) and (x1)) or ((x0) and (x2))) ^ (x0)) and (((x2) or (x0)) ^ (x0))) ^ ((x0) and (x2))) ^ ((x0) and (x2)))) and ((x2) or (x0))) or (x0))
    

@classical_function
def cf_v3_c11(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((((not ((((((not (x1)) or (not (x0))) ^ (not (x0))) or (x0)) or ((x2) or (x2))) ^ ((x2) or (x2)))) and (not ((x2) ^ (x1)))) ^ (x1)) ^ (x1)) or ((x0) ^ (x2))
    
def pf_v3_c11(x0, x1, x2):
    return ((((not ((((((not (x1)) or (not (x0))) ^ (not (x0))) or (x0)) or ((x2) or (x2))) ^ ((x2) or (x2)))) and (not ((x2) ^ (x1)))) ^ (x1)) ^ (x1)) or ((x0) ^ (x2))
    

@classical_function
def cf_v3_c12(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return (not (not (((((not (((not ((x1) and (x2))) and (not (x0))) or (x1))) ^ (x0)) ^ ((not (x1)) ^ (x2))) or (x2)) ^ ((x2) and (not (x0)))))) ^ ((not (x0)) or (x1))
    
def pf_v3_c12(x0, x1, x2):
    return (not (not (((((not (((not ((x1) and (x2))) and (not (x0))) or (x1))) ^ (x0)) ^ ((not (x1)) ^ (x2))) or (x2)) ^ ((x2) and (not (x0)))))) ^ ((not (x0)) or (x1))
    

@classical_function
def cf_v3_c13(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not (not ((((((not ((((((x2) ^ (x0)) or (x1)) or (x0)) ^ ((x0) ^ (x2))) and ((x2) or (x0)))) ^ (((x2) or (x0)) or (x2))) ^ (x1)) or (not ((x0) ^ (x2)))) ^ (not ((x0) ^ (x2)))) ^ (x2)))
    
def pf_v3_c13(x0, x1, x2):
    return not (not ((((((not ((((((x2) ^ (x0)) or (x1)) or (x0)) ^ ((x0) ^ (x2))) and ((x2) or (x0)))) ^ (((x2) or (x0)) or (x2))) ^ (x1)) or (not ((x0) ^ (x2)))) ^ (not ((x0) ^ (x2)))) ^ (x2)))
    

@classical_function
def cf_v3_c14(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not ((not (not (((not (((not (((((x0) or (x2)) ^ (x2)) ^ (x0)) and (x1))) ^ ((x0) ^ (x1))) ^ (x0))) and (not ((x2) or (x0)))) or (x0)))) ^ (x1))
    
def pf_v3_c14(x0, x1, x2):
    return not ((not (not (((not (((not (((((x0) or (x2)) ^ (x2)) ^ (x0)) and (x1))) ^ ((x0) ^ (x1))) ^ (x0))) and (not ((x2) or (x0)))) or (x0)))) ^ (x1))
    

@classical_function
def cf_v3_c15(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return (not ((not ((((not ((((((not ((x1) ^ (x1))) ^ (x0)) or (not (x0))) or (x1)) and (x2)) or (x0))) ^ (x0)) ^ (x2)) ^ ((x0) ^ (not (x0))))) ^ ((x1) ^ (x1)))) and ((x0) or ((x1) or (not (x0))))
    
def pf_v3_c15(x0, x1, x2):
    return (not ((not ((((not ((((((not ((x1) ^ (x1))) ^ (x0)) or (not (x0))) or (x1)) and (x2)) or (x0))) ^ (x0)) ^ (x2)) ^ ((x0) ^ (not (x0))))) ^ ((x1) ^ (x1)))) and ((x0) or ((x1) or (not (x0))))
    

@classical_function
def cf_v3_c16(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return ((not ((((((((((not ((not (not (x1))) ^ (x2))) ^ ((not (x0)) ^ (not (not (x0))))) and (not (x0))) ^ (x0)) or (not (not (x0)))) and (not (x2))) ^ (not (x2))) or (((not (x0)) ^ (not (not (x0)))) ^ (x0))) ^ ((x0) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((x0) ^ (x1))) ^ (((not (x0)) ^ (not (not (x0)))) ^ (x0))
    
def pf_v3_c16(x0, x1, x2):
    return ((not ((((((((((not ((not (not (x1))) ^ (x2))) ^ ((not (x0)) ^ (not (not (x0))))) and (not (x0))) ^ (x0)) or (not (not (x0)))) and (not (x2))) ^ (not (x2))) or (((not (x0)) ^ (not (not (x0)))) ^ (x0))) ^ ((x0) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((x0) ^ (x1))) ^ (((not (x0)) ^ (not (not (x0)))) ^ (x0))
    

@classical_function
def cf_v3_c17(x0 : Int1, x1 : Int1, x2 : Int1) -> Int1:
    return not ((((((not (((((not (((not ((not (x0)) ^ (x0))) or (x1)) or (x0))) ^ (x1)) and ((x2) ^ (x1))) and ((x2) or (x0))) ^ ((not (not (x1))) or (x2)))) and (not (not (x1)))) ^ (x2)) and ((x2) or (x0))) or (not (x1))) ^ ((x0) and ((x2) ^ (x1))))
    
def pf_v3_c17(x0, x1, x2):
    return not ((((((not (((((not (((not ((not (x0)) ^ (x0))) or (x1)) or (x0))) ^ (x1)) and ((x2) ^ (x1))) and ((x2) or (x0))) ^ ((not (not (x1))) or (x2)))) and (not (not (x1)))) ^ (x2)) and ((x2) or (x0))) or (not (x1))) ^ ((x0) and ((x2) ^ (x1))))
    

@classical_function
def cf_v4_c3(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (not ((x3) ^ (x0))) ^ (not ((x1) ^ (x2)))
    
def pf_v4_c3(x0, x1, x2, x3):
    return (not ((x3) ^ (x0))) ^ (not ((x1) ^ (x2)))
    

@classical_function
def cf_v4_c4(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (not (((x0) or (x3)) and ((x1) or (x0)))) ^ (x2)
    
def pf_v4_c4(x0, x1, x2, x3):
    return (not (((x0) or (x3)) and ((x1) or (x0)))) ^ (x2)
    

@classical_function
def cf_v4_c5(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return ((((not (x1)) and (x2)) ^ (x2)) ^ (x3)) and (x0)
    
def pf_v4_c5(x0, x1, x2, x3):
    return ((((not (x1)) and (x2)) ^ (x2)) ^ (x3)) and (x0)
    

@classical_function
def cf_v4_c6(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return ((((((x3) or (x1)) and ((x3) or (x2))) or ((x0) and (x2))) ^ (x2)) or ((x3) or (x2))) ^ (x1)
    
def pf_v4_c6(x0, x1, x2, x3):
    return ((((((x3) or (x1)) and ((x3) or (x2))) or ((x0) and (x2))) ^ (x2)) or ((x3) or (x2))) ^ (x1)
    

@classical_function
def cf_v4_c7(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (((((not ((x0) and (x0))) ^ (x2)) ^ (((x2) and (x3)) ^ ((x2) and (x3)))) and ((x2) and (x3))) or (x1)) and (x2)
    
def pf_v4_c7(x0, x1, x2, x3):
    return (((((not ((x0) and (x0))) ^ (x2)) ^ (((x2) and (x3)) ^ ((x2) and (x3)))) and ((x2) and (x3))) or (x1)) and (x2)
    

@classical_function
def cf_v4_c8(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return not (((((((not (x2)) and (x0)) and (x0)) ^ ((not (x1)) and (x3))) ^ (x2)) ^ (x1)) and (x0))
    
def pf_v4_c8(x0, x1, x2, x3):
    return not (((((((not (x2)) and (x0)) and (x0)) ^ ((not (x1)) and (x3))) ^ (x2)) ^ (x1)) and (x0))
    

@classical_function
def cf_v4_c9(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return ((((not (((((x3) and (x2)) or ((x3) and (x0))) ^ (x1)) ^ (x0))) and (x1)) ^ (((x3) and (x0)) ^ ((x3) and (x0)))) or (x1)) and (x0)
    
def pf_v4_c9(x0, x1, x2, x3):
    return ((((not (((((x3) and (x2)) or ((x3) and (x0))) ^ (x1)) ^ (x0))) and (x1)) ^ (((x3) and (x0)) ^ ((x3) and (x0)))) or (x1)) and (x0)
    

@classical_function
def cf_v4_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return not (((((not (((((x0) ^ (x2)) ^ (x2)) or (x3)) and ((x3) ^ (x0)))) and ((x1) or (x3))) and (x3)) and (not ((x1) or (x3)))) ^ (((x2) ^ (x1)) and ((x3) ^ (x0))))
    
def pf_v4_c10(x0, x1, x2, x3):
    return not (((((not (((((x0) ^ (x2)) ^ (x2)) or (x3)) and ((x3) ^ (x0)))) and ((x1) or (x3))) and (x3)) and (not ((x1) or (x3)))) ^ (((x2) ^ (x1)) and ((x3) ^ (x0))))
    

@classical_function
def cf_v4_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return ((((not (((((((x1) ^ (x1)) and (x3)) ^ (x3)) and (x2)) and ((x0) and ((x0) ^ (x1)))) ^ (x0))) or ((x0) ^ (x1))) ^ (not (((x1) ^ (x1)) ^ ((x1) and (x0))))) ^ ((x1) and (x0))) or (((x1) and (x0)) and ((x0) and ((x0) ^ (x1))))
    
def pf_v4_c11(x0, x1, x2, x3):
    return ((((not (((((((x1) ^ (x1)) and (x3)) ^ (x3)) and (x2)) and ((x0) and ((x0) ^ (x1)))) ^ (x0))) or ((x0) ^ (x1))) ^ (not (((x1) ^ (x1)) ^ ((x1) and (x0))))) ^ ((x1) and (x0))) or (((x1) and (x0)) and ((x0) and ((x0) ^ (x1))))
    

@classical_function
def cf_v4_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (((not (((((((((x0) ^ (x3)) ^ (x1)) ^ (x3)) or (((x3) ^ (x0)) ^ (x1))) and (((x3) ^ (x0)) ^ (x1))) ^ (x1)) and ((x0) and (x2))) or ((x1) ^ (((x3) ^ (x0)) ^ (x1))))) or (x2)) and ((x1) ^ (((x3) ^ (x0)) ^ (x1)))) ^ ((x0) and (x2))
    
def pf_v4_c12(x0, x1, x2, x3):
    return (((not (((((((((x0) ^ (x3)) ^ (x1)) ^ (x3)) or (((x3) ^ (x0)) ^ (x1))) and (((x3) ^ (x0)) ^ (x1))) ^ (x1)) and ((x0) and (x2))) or ((x1) ^ (((x3) ^ (x0)) ^ (x1))))) or (x2)) and ((x1) ^ (((x3) ^ (x0)) ^ (x1)))) ^ ((x0) and (x2))
    

@classical_function
def cf_v4_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return ((((((not (not ((((((x0) or (x2)) and (x2)) or (x2)) and (x2)) or ((x2) or (x2))))) or (x3)) or (not (x1))) ^ (not ((x1) and (x2)))) or ((x0) or (((x2) or (x2)) or (x3)))) ^ ((x0) or (x3))) ^ ((x0) or (x3))
    
def pf_v4_c13(x0, x1, x2, x3):
    return ((((((not (not ((((((x0) or (x2)) and (x2)) or (x2)) and (x2)) or ((x2) or (x2))))) or (x3)) or (not (x1))) ^ (not ((x1) and (x2)))) or ((x0) or (((x2) or (x2)) or (x3)))) ^ ((x0) or (x3))) ^ ((x0) or (x3))
    

@classical_function
def cf_v4_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (not ((((((((((((not (x1)) or (not (x3))) ^ (x0)) and (((x3) or (x0)) ^ (x2))) or (not (x3))) and (x2)) and (x3)) and ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) or (x1)) or (((x3) or (x0)) and (x2))) ^ ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) and ((x0) or ((((x3) or (x0)) ^ (x2)) or ((x2) and ((x3) or (x0))))))) or (x2)
    
def pf_v4_c14(x0, x1, x2, x3):
    return (not ((((((((((((not (x1)) or (not (x3))) ^ (x0)) and (((x3) or (x0)) ^ (x2))) or (not (x3))) and (x2)) and (x3)) and ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) or (x1)) or (((x3) or (x0)) and (x2))) ^ ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) and ((x0) or ((((x3) or (x0)) ^ (x2)) or ((x2) and ((x3) or (x0))))))) or (x2)
    

@classical_function
def cf_v4_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (((not ((not ((((not ((not ((((x0) ^ (x1)) and ((x3) ^ (x0))) ^ (x2))) and ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (x0))) and (x3))) and (x3)) or (not (x0))) or (x0)
    
def pf_v4_c15(x0, x1, x2, x3):
    return (((not ((not ((((not ((not ((((x0) ^ (x1)) and ((x3) ^ (x0))) ^ (x2))) and ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (x0))) and (x3))) and (x3)) or (not (x0))) or (x0)
    

@classical_function
def cf_v4_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (((((((not ((not (not (((not (((x3) ^ (x0)) and (x0))) or (x1)) and ((x2) and (x1))))) and (((x1) ^ (x3)) or ((x1) ^ (x3))))) and (((x1) ^ (x3)) or ((x1) ^ (x3)))) and ((not ((x1) ^ (x3))) and ((x2) and (x1)))) or ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not (x1))) ^ ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not ((x1) ^ (x3)))) or (((x1) ^ (x3)) or ((x1) ^ (x3)))
    
def pf_v4_c16(x0, x1, x2, x3):
    return (((((((not ((not (not (((not (((x3) ^ (x0)) and (x0))) or (x1)) and ((x2) and (x1))))) and (((x1) ^ (x3)) or ((x1) ^ (x3))))) and (((x1) ^ (x3)) or ((x1) ^ (x3)))) and ((not ((x1) ^ (x3))) and ((x2) and (x1)))) or ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not (x1))) ^ ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not ((x1) ^ (x3)))) or (((x1) ^ (x3)) or ((x1) ^ (x3)))
    

@classical_function
def cf_v4_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (((((((not (((((((((not (x1)) and (not (x1))) or (x2)) and ((x2) or (x1))) or (x3)) ^ ((x0) and (x2))) and ((x1) and (x3))) or (x3)) ^ (x3))) or ((not (x1)) and ((x2) or (x1)))) and (x0)) or (x3)) or ((not (x1)) and ((x2) or (x1)))) ^ (((x3) ^ ((x1) and (x3))) or (not (x1)))) or ((x1) and (x3))) ^ (x3)
    
def pf_v4_c17(x0, x1, x2, x3):
    return (((((((not (((((((((not (x1)) and (not (x1))) or (x2)) and ((x2) or (x1))) or (x3)) ^ ((x0) and (x2))) and ((x1) and (x3))) or (x3)) ^ (x3))) or ((not (x1)) and ((x2) or (x1)))) and (x0)) or (x3)) or ((not (x1)) and ((x2) or (x1)))) ^ (((x3) ^ ((x1) and (x3))) or (not (x1)))) or ((x1) and (x3))) ^ (x3)
    

@classical_function
def cf_v4_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1) -> Int1:
    return (not (((((not (((not (((not ((((not (not (x0))) and (x3)) ^ (not (x0))) ^ (x3))) and (x0)) ^ (not (not (x0))))) and ((not (x0)) ^ (x0))) or (x1))) ^ (not (x0))) and ((x2) and (x3))) and ((not (x0)) ^ (not (x0)))) ^ ((x2) and (x3)))) and (x3)
    
def pf_v4_c18(x0, x1, x2, x3):
    return (not (((((not (((not (((not ((((not (not (x0))) and (x3)) ^ (not (x0))) ^ (x3))) and (x0)) ^ (not (not (x0))))) and ((not (x0)) ^ (x0))) or (x1))) ^ (not (x0))) and ((x2) and (x3))) and ((not (x0)) ^ (not (x0)))) ^ ((x2) and (x3)))) and (x3)
    

@classical_function
def cf_v5_c4(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((((x3) and (x1)) and (x0)) ^ ((x1) and ((x4) and (x1)))) and ((x2) ^ (x2))
    
def pf_v5_c4(x0, x1, x2, x3, x4):
    return ((((x3) and (x1)) and (x0)) ^ ((x1) and ((x4) and (x1)))) and ((x2) ^ (x2))
    

@classical_function
def cf_v5_c5(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((((x2) ^ (x0)) or (x0)) ^ (x4)) ^ ((x1) ^ (x3))) ^ (((x1) ^ (x3)) or (x0))
    
def pf_v5_c5(x0, x1, x2, x3, x4):
    return (((((x2) ^ (x0)) or (x0)) ^ (x4)) ^ ((x1) ^ (x3))) ^ (((x1) ^ (x3)) or (x0))
    

@classical_function
def cf_v5_c6(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((((((x3) ^ (x2)) or (x0)) or (x3)) ^ ((x3) or (x3))) ^ ((x4) ^ ((x3) or (x3)))) and (x1)
    
def pf_v5_c6(x0, x1, x2, x3, x4):
    return ((((((x3) ^ (x2)) or (x0)) or (x3)) ^ ((x3) or (x3))) ^ ((x4) ^ ((x3) or (x3)))) and (x1)
    

@classical_function
def cf_v5_c7(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((not (((not ((x3) and (x3))) ^ (not (x2))) and ((x4) and (x2)))) and (x1)) and (x0)
    
def pf_v5_c7(x0, x1, x2, x3, x4):
    return ((not (((not ((x3) and (x3))) ^ (not (x2))) and ((x4) and (x2)))) and (x1)) and (x0)
    

@classical_function
def cf_v5_c8(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return not ((((((((x4) or (x4)) ^ (x2)) or (x3)) and (x2)) or ((x4) and (x0))) or (((x0) ^ (x2)) or (x4))) or (((x2) or (x1)) or (x0)))
    
def pf_v5_c8(x0, x1, x2, x3, x4):
    return not ((((((((x4) or (x4)) ^ (x2)) or (x3)) and (x2)) or ((x4) and (x0))) or (((x0) ^ (x2)) or (x4))) or (((x2) or (x1)) or (x0)))
    

@classical_function
def cf_v5_c9(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((((((not ((x4) or (x3))) ^ (x0)) and (x2)) ^ (not (x1))) ^ (x2)) and (x4)) or (x4)) ^ (((x0) ^ (x0)) and (x2))
    
def pf_v5_c9(x0, x1, x2, x3, x4):
    return (((((((not ((x4) or (x3))) ^ (x0)) and (x2)) ^ (not (x1))) ^ (x2)) and (x4)) or (x4)) ^ (((x0) ^ (x0)) and (x2))
    

@classical_function
def cf_v5_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((((((not ((not ((x3) ^ (x4))) or (not (x0)))) or (x2)) and ((x1) ^ (x4))) ^ (x2)) and (x1)) ^ ((x0) or (x4))) or (not (x0))
    
def pf_v5_c10(x0, x1, x2, x3, x4):
    return ((((((not ((not ((x3) ^ (x4))) or (not (x0)))) or (x2)) and ((x1) ^ (x4))) ^ (x2)) and (x1)) ^ ((x0) or (x4))) or (not (x0))
    

@classical_function
def cf_v5_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((((not (((not (((x0) and (x1)) or (x4))) and (x0)) and ((x2) or ((x2) and (x3))))) or (x4)) ^ ((x1) or (x4))) and ((x2) and (x3))) and (((x1) and (x2)) ^ (x2))) and ((x2) or ((x2) and (x3)))
    
def pf_v5_c11(x0, x1, x2, x3, x4):
    return (((((not (((not (((x0) and (x1)) or (x4))) and (x0)) and ((x2) or ((x2) and (x3))))) or (x4)) ^ ((x1) or (x4))) and ((x2) and (x3))) and (((x1) and (x2)) ^ (x2))) and ((x2) or ((x2) and (x3)))
    

@classical_function
def cf_v5_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((((((((not (((x3) and (x1)) or (x3))) ^ ((x2) and (x3))) ^ (((x2) and (x3)) ^ (x1))) or (x4)) and (((x2) and (x3)) ^ (x1))) or (x1)) and ((x1) or (x0))) ^ (((x1) or (x2)) and (x0))) or (x1)) or (x0)
    
def pf_v5_c12(x0, x1, x2, x3, x4):
    return (((((((((not (((x3) and (x1)) or (x3))) ^ ((x2) and (x3))) ^ (((x2) and (x3)) ^ (x1))) or (x4)) and (((x2) and (x3)) ^ (x1))) or (x1)) and ((x1) or (x0))) ^ (((x1) or (x2)) and (x0))) or (x1)) or (x0)
    

@classical_function
def cf_v5_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((((((((((not (((x2) ^ (x2)) ^ (x1))) ^ (x3)) or ((x4) ^ (x0))) ^ (x0)) and (x2)) and (x1)) or ((x0) and ((x4) ^ (x0)))) ^ ((x2) and (x2))) and ((x2) or (x1))) ^ (((x2) or (x1)) ^ (((x4) ^ (x0)) ^ ((x4) ^ (x0))))) and (x2)
    
def pf_v5_c13(x0, x1, x2, x3, x4):
    return ((((((((((not (((x2) ^ (x2)) ^ (x1))) ^ (x3)) or ((x4) ^ (x0))) ^ (x0)) and (x2)) and (x1)) or ((x0) and ((x4) ^ (x0)))) ^ ((x2) and (x2))) and ((x2) or (x1))) ^ (((x2) or (x1)) ^ (((x4) ^ (x0)) ^ ((x4) ^ (x0))))) and (x2)
    

@classical_function
def cf_v5_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return not (((((not ((((((((not (x2)) or (not (x0))) and (x2)) and (not (x0))) or (x4)) or ((x1) or (x0))) and (x0)) and (x4))) or ((x2) and ((x1) or (x0)))) ^ (((x2) and ((x1) or (x0))) or ((x3) and (not (x0))))) ^ (x0)) and (x3))
    
def pf_v5_c14(x0, x1, x2, x3, x4):
    return not (((((not ((((((((not (x2)) or (not (x0))) and (x2)) and (not (x0))) or (x4)) or ((x1) or (x0))) and (x0)) and (x4))) or ((x2) and ((x1) or (x0)))) ^ (((x2) and ((x1) or (x0))) or ((x3) and (not (x0))))) ^ (x0)) and (x3))
    

@classical_function
def cf_v5_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((not (not (((not (not ((((((not (x0)) ^ (x2)) or (x4)) or ((x4) ^ (x3))) or (x4)) or ((x1) or (x2))))) or (not ((x3) or ((x1) or (x2))))) ^ ((x1) or (x2))))) and (not ((not (x0)) or (x4)))) and (((x4) ^ (x3)) and (x2))) ^ ((x4) or ((x4) ^ (x3)))
    
def pf_v5_c15(x0, x1, x2, x3, x4):
    return (((not (not (((not (not ((((((not (x0)) ^ (x2)) or (x4)) or ((x4) ^ (x3))) or (x4)) or ((x1) or (x2))))) or (not ((x3) or ((x1) or (x2))))) ^ ((x1) or (x2))))) and (not ((not (x0)) or (x4)))) and (((x4) ^ (x3)) and (x2))) ^ ((x4) or ((x4) ^ (x3)))
    

@classical_function
def cf_v5_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((not (((not ((((((((not (((x1) and (x4)) and (x2))) ^ (not ((x1) and (x0)))) ^ (x3)) ^ ((x0) and (x0))) ^ (x0)) or ((x3) ^ (x3))) ^ ((x0) and (x0))) or (x0))) ^ (x4)) or ((not ((x1) and (x0))) ^ (x0)))) ^ (x3)) and (x3)
    
def pf_v5_c16(x0, x1, x2, x3, x4):
    return ((not (((not ((((((((not (((x1) and (x4)) and (x2))) ^ (not ((x1) and (x0)))) ^ (x3)) ^ ((x0) and (x0))) ^ (x0)) or ((x3) ^ (x3))) ^ ((x0) and (x0))) or (x0))) ^ (x4)) or ((not ((x1) and (x0))) ^ (x0)))) ^ (x3)) and (x3)
    

@classical_function
def cf_v5_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return not ((((((((not ((((((not (((x2) and (x0)) ^ ((x1) and (x4)))) and (x0)) or ((x1) and (x4))) and (x2)) ^ (x0)) and (not (x0)))) or ((x0) ^ (x3))) or ((not (x0)) and (x4))) ^ ((x4) ^ (x3))) and (x3)) or ((x0) ^ (x3))) or (x4)) and ((x0) and (((not (x0)) or (x0)) and ((x1) and (x4)))))
    
def pf_v5_c17(x0, x1, x2, x3, x4):
    return not ((((((((not ((((((not (((x2) and (x0)) ^ ((x1) and (x4)))) and (x0)) or ((x1) and (x4))) and (x2)) ^ (x0)) and (not (x0)))) or ((x0) ^ (x3))) or ((not (x0)) and (x4))) ^ ((x4) ^ (x3))) and (x3)) or ((x0) ^ (x3))) or (x4)) and ((x0) and (((not (x0)) or (x0)) and ((x1) and (x4)))))
    

@classical_function
def cf_v5_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return (((not ((not (not (not ((not ((((((not ((not (x4)) or (x4))) ^ (x1)) ^ (x3)) ^ (x2)) ^ (not (x4))) and (x2))) and (((x4) or (x4)) ^ (x1)))))) ^ (not (x2)))) and (not ((((x4) or (x4)) ^ (x1)) and (x0)))) or ((x4) or (x4))) or ((not (x2)) and (not (x4)))
    
def pf_v5_c18(x0, x1, x2, x3, x4):
    return (((not ((not (not (not ((not ((((((not ((not (x4)) or (x4))) ^ (x1)) ^ (x3)) ^ (x2)) ^ (not (x4))) and (x2))) and (((x4) or (x4)) ^ (x1)))))) ^ (not (x2)))) and (not ((((x4) or (x4)) ^ (x1)) and (x0)))) or ((x4) or (x4))) or ((not (x2)) and (not (x4)))
    

@classical_function
def cf_v5_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1) -> Int1:
    return ((((not ((((not ((((((((not (not ((x3) and (x2)))) and (x2)) and (not (x2))) and (x3)) or ((x4) and (not (x4)))) and (not (x4))) ^ (x0)) and ((x0) and (x1)))) and (((x0) and (x1)) and ((x4) and (not (x4))))) and ((x0) and (x1))) or ((not (x2)) and (x1)))) and (x3)) or (((x0) and (x1)) and ((x4) and (not (x4))))) and (((x4) and (not (x4))) ^ ((x3) and (x1)))) or ((x3) and (x1))
    
def pf_v5_c19(x0, x1, x2, x3, x4):
    return ((((not ((((not ((((((((not (not ((x3) and (x2)))) and (x2)) and (not (x2))) and (x3)) or ((x4) and (not (x4)))) and (not (x4))) ^ (x0)) and ((x0) and (x1)))) and (((x0) and (x1)) and ((x4) and (not (x4))))) and ((x0) and (x1))) or ((not (x2)) and (x1)))) and (x3)) or (((x0) and (x1)) and ((x4) and (not (x4))))) and (((x4) and (not (x4))) ^ ((x3) and (x1)))) or ((x3) and (x1))
    

@classical_function
def cf_v6_c6(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return ((((((x4) ^ (x0)) or (x1)) ^ (x5)) ^ (x4)) ^ ((x1) or ((x2) ^ (x3)))) and (x5)
    
def pf_v6_c6(x0, x1, x2, x3, x4, x5):
    return ((((((x4) ^ (x0)) or (x1)) ^ (x5)) ^ (x4)) ^ ((x1) or ((x2) ^ (x3)))) and (x5)
    

@classical_function
def cf_v6_c7(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (not ((((((x2) and (x0)) and (x1)) or (x4)) or ((x0) or ((x3) and (x5)))) ^ (x3))) ^ ((x0) or ((x3) and (x5)))
    
def pf_v6_c7(x0, x1, x2, x3, x4, x5):
    return (not ((((((x2) and (x0)) and (x1)) or (x4)) or ((x0) or ((x3) and (x5)))) ^ (x3))) ^ ((x0) or ((x3) and (x5)))
    

@classical_function
def cf_v6_c8(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return not (((((not (((x3) ^ (x1)) and (x0))) and (x1)) and (x4)) or ((x1) and (x5))) and (x2))
    
def pf_v6_c8(x0, x1, x2, x3, x4, x5):
    return not (((((not (((x3) ^ (x1)) and (x0))) and (x1)) and (x4)) or ((x1) and (x5))) and (x2))
    

@classical_function
def cf_v6_c9(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (((((((not ((x3) or (x5))) or (not (x2))) ^ (x1)) ^ (x3)) or ((x4) ^ ((x0) or (x4)))) or (x0)) and (not (x2))) ^ (x2)
    
def pf_v6_c9(x0, x1, x2, x3, x4, x5):
    return (((((((not ((x3) or (x5))) or (not (x2))) ^ (x1)) ^ (x3)) or ((x4) ^ ((x0) or (x4)))) or (x0)) and (not (x2))) ^ (x2)
    

@classical_function
def cf_v6_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return ((((((((((x1) ^ (x0)) or (x1)) or (x0)) and (x5)) and (x3)) ^ (x4)) or (x4)) ^ ((x5) ^ (x1))) and ((x4) or (x2))) and (x0)
    
def pf_v6_c10(x0, x1, x2, x3, x4, x5):
    return ((((((((((x1) ^ (x0)) or (x1)) or (x0)) and (x5)) and (x3)) ^ (x4)) or (x4)) ^ ((x5) ^ (x1))) and ((x4) or (x2))) and (x0)
    

@classical_function
def cf_v6_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (not (((((not (not ((((x5) or (x1)) ^ (x2)) and (x1)))) or (((x3) ^ ((x4) or (x4))) and (x4))) and (x0)) and ((((x3) ^ ((x4) or (x4))) and (x4)) or (x0))) or (((x4) and (x0)) and (x3)))) and (not (x2))
    
def pf_v6_c11(x0, x1, x2, x3, x4, x5):
    return (not (((((not (not ((((x5) or (x1)) ^ (x2)) and (x1)))) or (((x3) ^ ((x4) or (x4))) and (x4))) and (x0)) and ((((x3) ^ ((x4) or (x4))) and (x4)) or (x0))) or (((x4) and (x0)) and (x3)))) and (not (x2))
    

@classical_function
def cf_v6_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (((not (((((not ((((x2) ^ (x5)) or (x1)) or (x5))) or (x2)) or (x4)) ^ (not (x5))) and (x4))) or (((x0) ^ (x5)) or (x4))) or (x3)) ^ (((x0) ^ (x5)) or (x4))
    
def pf_v6_c12(x0, x1, x2, x3, x4, x5):
    return (((not (((((not ((((x2) ^ (x5)) or (x1)) or (x5))) or (x2)) or (x4)) ^ (not (x5))) and (x4))) or (((x0) ^ (x5)) or (x4))) or (x3)) ^ (((x0) ^ (x5)) or (x4))
    

@classical_function
def cf_v6_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return not (((((((((((((x5) and (x1)) ^ (x0)) and (x2)) or ((x0) and (x1))) ^ (x0)) or (x2)) ^ ((x0) and (x1))) ^ ((x3) or (x1))) and ((x0) and (x1))) ^ (x4)) and ((x4) and (((x0) and (x1)) and (x2)))) or ((x4) and (((x0) and (x1)) and (x2))))
    
def pf_v6_c13(x0, x1, x2, x3, x4, x5):
    return not (((((((((((((x5) and (x1)) ^ (x0)) and (x2)) or ((x0) and (x1))) ^ (x0)) or (x2)) ^ ((x0) and (x1))) ^ ((x3) or (x1))) and ((x0) and (x1))) ^ (x4)) and ((x4) and (((x0) and (x1)) and (x2)))) or ((x4) and (((x0) and (x1)) and (x2))))
    

@classical_function
def cf_v6_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (((((not ((not (not (not (((not ((x2) or (x2))) ^ ((x2) or (x4))) or (x5))))) ^ (not (x0)))) ^ (not (not ((x5) ^ (not ((x2) or (x4))))))) and (x3)) and (not (x0))) or ((x0) or (x1))) ^ (x0)
    
def pf_v6_c14(x0, x1, x2, x3, x4, x5):
    return (((((not ((not (not (not (((not ((x2) or (x2))) ^ ((x2) or (x4))) or (x5))))) ^ (not (x0)))) ^ (not (not ((x5) ^ (not ((x2) or (x4))))))) and (x3)) and (not (x0))) or ((x0) or (x1))) ^ (x0)
    

@classical_function
def cf_v6_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (((((((not (((not (((((x3) and (x3)) or (x2)) and (x5)) or (x1))) or (x3)) and (x4))) or (x1)) and (not (x2))) ^ (x0)) or ((x5) and (x4))) or (not (x2))) and ((x2) and (x4))) or (((x1) ^ ((x2) and (x4))) and (not ((x4) and (((x2) and (x4)) or (x0)))))
    
def pf_v6_c15(x0, x1, x2, x3, x4, x5):
    return (((((((not (((not (((((x3) and (x3)) or (x2)) and (x5)) or (x1))) or (x3)) and (x4))) or (x1)) and (not (x2))) ^ (x0)) or ((x5) and (x4))) or (not (x2))) and ((x2) and (x4))) or (((x1) ^ ((x2) and (x4))) and (not ((x4) and (((x2) and (x4)) or (x0)))))
    

@classical_function
def cf_v6_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (not (((((((not (((((not (not ((x4) and (x3)))) or ((x1) and (x1))) ^ (x2)) and (x3)) and (not (x0)))) and (x4)) ^ (not (x0))) and (((x2) ^ (x3)) and ((x1) and (x1)))) ^ (((not (x2)) or (not (x2))) ^ (x0))) ^ (x5)) or (x0))) and (x5)
    
def pf_v6_c16(x0, x1, x2, x3, x4, x5):
    return (not (((((((not (((((not (not ((x4) and (x3)))) or ((x1) and (x1))) ^ (x2)) and (x3)) and (not (x0)))) and (x4)) ^ (not (x0))) and (((x2) ^ (x3)) and ((x1) and (x1)))) ^ (((not (x2)) or (not (x2))) ^ (x0))) ^ (x5)) or (x0))) and (x5)
    

@classical_function
def cf_v6_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return ((((((not (((((not ((not ((((x2) ^ (x0)) ^ (x1)) ^ (x2))) or ((x5) ^ (x3)))) and (x0)) and (x4)) or ((x0) ^ (x5))) ^ (x3))) ^ (x0)) or (x0)) ^ (not (x2))) ^ (((x5) and (not (x1))) and (x5))) or (x2)) or ((((x5) and (not (x1))) and (x5)) ^ (x0))
    
def pf_v6_c17(x0, x1, x2, x3, x4, x5):
    return ((((((not (((((not ((not ((((x2) ^ (x0)) ^ (x1)) ^ (x2))) or ((x5) ^ (x3)))) and (x0)) and (x4)) or ((x0) ^ (x5))) ^ (x3))) ^ (x0)) or (x0)) ^ (not (x2))) ^ (((x5) and (not (x1))) and (x5))) or (x2)) or ((((x5) and (not (x1))) and (x5)) ^ (x0))
    

@classical_function
def cf_v6_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (not (((((not (((((((not (((not ((x1) ^ (x0))) or (x0)) or (x2))) or (x0)) and ((x4) ^ (x4))) and (x0)) or (not (x3))) or (((not (not (x3))) and (x3)) or (x1))) and ((x4) ^ (x4)))) or ((x1) and (x5))) or (((x1) and (((x4) ^ (x4)) or (x5))) or (((x4) ^ (x4)) or (x5)))) and (x1)) or (x2))) ^ (x5)
    
def pf_v6_c18(x0, x1, x2, x3, x4, x5):
    return (not (((((not (((((((not (((not ((x1) ^ (x0))) or (x0)) or (x2))) or (x0)) and ((x4) ^ (x4))) and (x0)) or (not (x3))) or (((not (not (x3))) and (x3)) or (x1))) and ((x4) ^ (x4)))) or ((x1) and (x5))) or (((x1) and (((x4) ^ (x4)) or (x5))) or (((x4) ^ (x4)) or (x5)))) and (x1)) or (x2))) ^ (x5)
    

@classical_function
def cf_v6_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return not (not (not ((((((not ((((not ((not ((not (((x0) ^ (x0)) ^ ((x3) ^ (x2)))) ^ (x0))) ^ ((x1) ^ ((x3) ^ (x2))))) or (x1)) or (x2)) ^ (x5))) ^ (x4)) ^ ((x3) ^ (x2))) and ((not ((x1) ^ ((x3) ^ (x2)))) or (not (not ((x1) ^ ((x3) ^ (x2))))))) ^ (not (x4))) ^ ((x2) ^ (not (x4))))))
    
def pf_v6_c19(x0, x1, x2, x3, x4, x5):
    return not (not (not ((((((not ((((not ((not ((not (((x0) ^ (x0)) ^ ((x3) ^ (x2)))) ^ (x0))) ^ ((x1) ^ ((x3) ^ (x2))))) or (x1)) or (x2)) ^ (x5))) ^ (x4)) ^ ((x3) ^ (x2))) and ((not ((x1) ^ ((x3) ^ (x2)))) or (not (not ((x1) ^ ((x3) ^ (x2))))))) ^ (not (x4))) ^ ((x2) ^ (not (x4))))))
    

@classical_function
def cf_v6_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1) -> Int1:
    return (not ((((((not ((((not ((not (not ((((((x0) ^ (x5)) and (x5)) or ((x5) ^ (x1))) or (x1)) and ((x3) and (x2))))) and (x4))) and (x0)) or (x5)) ^ (x2))) ^ ((x4) or ((x5) ^ (x1)))) ^ ((x4) or ((x5) ^ (x1)))) and (x2)) ^ ((x0) and ((x4) or ((x5) ^ (x1))))) ^ (x3))) ^ ((x4) and ((x3) and (x2)))
    
def pf_v6_c20(x0, x1, x2, x3, x4, x5):
    return (not ((((((not ((((not ((not (not ((((((x0) ^ (x5)) and (x5)) or ((x5) ^ (x1))) or (x1)) and ((x3) and (x2))))) and (x4))) and (x0)) or (x5)) ^ (x2))) ^ ((x4) or ((x5) ^ (x1)))) ^ ((x4) or ((x5) ^ (x1)))) and (x2)) ^ ((x0) and ((x4) or ((x5) ^ (x1))))) ^ (x3))) ^ ((x4) and ((x3) and (x2)))
    

@classical_function
def cf_v7_c7(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return (not ((((((x6) ^ (x4)) and (x1)) and (x6)) ^ ((x6) ^ (x6))) or ((x2) and ((x0) and (x4))))) and ((x5) ^ (x3))
    
def pf_v7_c7(x0, x1, x2, x3, x4, x5, x6):
    return (not ((((((x6) ^ (x4)) and (x1)) and (x6)) ^ ((x6) ^ (x6))) or ((x2) and ((x0) and (x4))))) and ((x5) ^ (x3))
    

@classical_function
def cf_v7_c8(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((((((not ((x4) ^ (x0))) and (x2)) ^ (x1)) and (x5)) ^ (((x4) and (x3)) ^ (x6))) or (x2)) ^ (((x4) and (x3)) ^ (x6))
    
def pf_v7_c8(x0, x1, x2, x3, x4, x5, x6):
    return ((((((not ((x4) ^ (x0))) and (x2)) ^ (x1)) and (x5)) ^ (((x4) and (x3)) ^ (x6))) or (x2)) ^ (((x4) and (x3)) ^ (x6))
    

@classical_function
def cf_v7_c9(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return (((((not ((((x0) ^ (x1)) or (x5)) or ((x2) or (x5)))) ^ (x4)) ^ (not (x3))) and (((x2) or (x5)) ^ ((x5) or (x2)))) and (x4)) ^ (x6)
    
def pf_v7_c9(x0, x1, x2, x3, x4, x5, x6):
    return (((((not ((((x0) ^ (x1)) or (x5)) or ((x2) or (x5)))) ^ (x4)) ^ (not (x3))) and (((x2) or (x5)) ^ ((x5) or (x2)))) and (x4)) ^ (x6)
    

@classical_function
def cf_v7_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((((((not ((((x6) and (x0)) and (x4)) and (x2))) or (x0)) ^ ((x2) and (x6))) or ((x3) ^ (x1))) or (x4)) or (x3)) and ((x5) or ((x4) and (x5)))
    
def pf_v7_c10(x0, x1, x2, x3, x4, x5, x6):
    return ((((((not ((((x6) and (x0)) and (x4)) and (x2))) or (x0)) ^ ((x2) and (x6))) or ((x3) ^ (x1))) or (x4)) or (x3)) and ((x5) or ((x4) and (x5)))
    

@classical_function
def cf_v7_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return not (((((((not ((((x1) ^ (x2)) ^ (x3)) and (x4))) and (x1)) or ((x4) ^ (x5))) or (not (x1))) or ((x3) and (x2))) or (x6)) or ((x0) or (x5)))
    
def pf_v7_c11(x0, x1, x2, x3, x4, x5, x6):
    return not (((((((not ((((x1) ^ (x2)) ^ (x3)) and (x4))) and (x1)) or ((x4) ^ (x5))) or (not (x1))) or ((x3) and (x2))) or (x6)) or ((x0) or (x5)))
    

@classical_function
def cf_v7_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((not ((((((not ((((x2) ^ (x3)) or (x6)) and (x0))) ^ (((x4) ^ (x0)) or (x4))) or (x3)) or (x1)) or ((x2) or ((x4) ^ (x0)))) ^ (x5))) and (x4)) and ((x4) ^ (x0))
    
def pf_v7_c12(x0, x1, x2, x3, x4, x5, x6):
    return ((not ((((((not ((((x2) ^ (x3)) or (x6)) and (x0))) ^ (((x4) ^ (x0)) or (x4))) or (x3)) or (x1)) or ((x2) or ((x4) ^ (x0)))) ^ (x5))) and (x4)) and ((x4) ^ (x0))
    

@classical_function
def cf_v7_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return not (not ((not ((((((not ((((x5) or (x1)) and (x5)) ^ (x0))) and (x4)) ^ (x4)) or ((x5) or (x4))) and ((x6) ^ ((x3) and (x5)))) ^ (not (x2)))) ^ ((x6) ^ ((x3) and (x5)))))
    
def pf_v7_c13(x0, x1, x2, x3, x4, x5, x6):
    return not (not ((not ((((((not ((((x5) or (x1)) and (x5)) ^ (x0))) and (x4)) ^ (x4)) or ((x5) or (x4))) and ((x6) ^ ((x3) and (x5)))) ^ (not (x2)))) ^ ((x6) ^ ((x3) and (x5)))))
    

@classical_function
def cf_v7_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return not ((((((((((((not ((x1) or (x0))) ^ (x3)) or ((x2) or (x6))) and (x6)) or (x2)) ^ ((x6) or (x3))) ^ ((x4) ^ (not (x1)))) and ((x5) or (x4))) ^ (x2)) and ((x6) or (x3))) and ((x6) or (x3))) or ((x0) ^ (x6)))
    
def pf_v7_c14(x0, x1, x2, x3, x4, x5, x6):
    return not ((((((((((((not ((x1) or (x0))) ^ (x3)) or ((x2) or (x6))) and (x6)) or (x2)) ^ ((x6) or (x3))) ^ ((x4) ^ (not (x1)))) and ((x5) or (x4))) ^ (x2)) and ((x6) or (x3))) and ((x6) or (x3))) or ((x0) ^ (x6)))
    

@classical_function
def cf_v7_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((((not (((((((((((x2) or (x1)) ^ (x6)) or (x1)) or (x5)) ^ (x2)) ^ ((x6) or (x2))) and (x3)) ^ (x1)) ^ (x3)) or ((x0) ^ (((x4) or (x6)) ^ (x0))))) ^ (x6)) or ((x4) or (x6))) and (((x4) or (x6)) ^ (not ((x4) or (x6))))) ^ ((x5) or (not ((x4) or (x6))))
    
def pf_v7_c15(x0, x1, x2, x3, x4, x5, x6):
    return ((((not (((((((((((x2) or (x1)) ^ (x6)) or (x1)) or (x5)) ^ (x2)) ^ ((x6) or (x2))) and (x3)) ^ (x1)) ^ (x3)) or ((x0) ^ (((x4) or (x6)) ^ (x0))))) ^ (x6)) or ((x4) or (x6))) and (((x4) or (x6)) ^ (not ((x4) or (x6))))) ^ ((x5) or (not ((x4) or (x6))))
    

@classical_function
def cf_v7_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return (((((((not (not (((not (((((x1) and (x1)) ^ ((x0) and (x4))) and (x0)) and (x6))) or (x2)) or (x5)))) or (not (x5))) or (x2)) or (not (x5))) ^ ((((x0) and (x4)) ^ (x4)) or ((x5) and ((x0) and (x4))))) and ((x0) and (x4))) ^ (x3)) or (x1)
    
def pf_v7_c16(x0, x1, x2, x3, x4, x5, x6):
    return (((((((not (not (((not (((((x1) and (x1)) ^ ((x0) and (x4))) and (x0)) and (x6))) or (x2)) or (x5)))) or (not (x5))) or (x2)) or (not (x5))) ^ ((((x0) and (x4)) ^ (x4)) or ((x5) and ((x0) and (x4))))) and ((x0) and (x4))) ^ (x3)) or (x1)
    

@classical_function
def cf_v7_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return not (not ((not ((not ((not (not ((((not (((((x0) and (x1)) or ((x4) and (x5))) and (x2)) or (x5))) and (x3)) and ((x6) and (x3))) or (x0)))) and ((x4) and (x5)))) ^ ((x2) or (x4)))) and (not (not ((x4) and (x5))))))
    
def pf_v7_c17(x0, x1, x2, x3, x4, x5, x6):
    return not (not ((not ((not ((not (not ((((not (((((x0) and (x1)) or ((x4) and (x5))) and (x2)) or (x5))) and (x3)) and ((x6) and (x3))) or (x0)))) and ((x4) and (x5)))) ^ ((x2) or (x4)))) and (not (not ((x4) and (x5))))))
    

@classical_function
def cf_v7_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((((not (((((((((not (((not ((x0) ^ (x3))) or (x1)) and ((x6) ^ (x2)))) or (x1)) and (not (x1))) ^ (x1)) ^ (((x6) and (x6)) ^ ((x4) and (not (x1))))) and (((x6) and (x6)) ^ (x5))) or (x3)) and (((x6) and (x6)) ^ (x5))) and ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5))))) or ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5)))) and (not (x4))) or ((x1) and ((x6) and (x6)))) ^ (x2)
    
def pf_v7_c18(x0, x1, x2, x3, x4, x5, x6):
    return ((((not (((((((((not (((not ((x0) ^ (x3))) or (x1)) and ((x6) ^ (x2)))) or (x1)) and (not (x1))) ^ (x1)) ^ (((x6) and (x6)) ^ ((x4) and (not (x1))))) and (((x6) and (x6)) ^ (x5))) or (x3)) and (((x6) and (x6)) ^ (x5))) and ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5))))) or ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5)))) and (not (x4))) or ((x1) and ((x6) and (x6)))) ^ (x2)
    

@classical_function
def cf_v7_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return not ((((((not ((((((not (((((((x3) ^ (x2)) or ((x5) ^ (x3))) and (x4)) or (x6)) ^ ((x2) or (x3))) ^ (x0))) and (x6)) and (x0)) ^ (((x2) or (x3)) and ((x5) ^ (x3)))) and (not (x1))) or (x3))) ^ ((x2) or (x3))) ^ ((x4) and ((((x3) or (x1)) and (x2)) ^ (x5)))) and (not ((x5) ^ (x3)))) and (((x3) or (x1)) and (x2))) or ((x5) and (x3)))
    
def pf_v7_c19(x0, x1, x2, x3, x4, x5, x6):
    return not ((((((not ((((((not (((((((x3) ^ (x2)) or ((x5) ^ (x3))) and (x4)) or (x6)) ^ ((x2) or (x3))) ^ (x0))) and (x6)) and (x0)) ^ (((x2) or (x3)) and ((x5) ^ (x3)))) and (not (x1))) or (x3))) ^ ((x2) or (x3))) ^ ((x4) and ((((x3) or (x1)) and (x2)) ^ (x5)))) and (not ((x5) ^ (x3)))) and (((x3) or (x1)) and (x2))) or ((x5) and (x3)))
    

@classical_function
def cf_v7_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return (((not (((((((not ((((((not ((((x0) ^ (x0)) ^ (x5)) ^ (x0))) ^ (x2)) or (x3)) or (((x6) ^ (x2)) ^ (x6))) and ((x1) or (x5))) or (x3))) ^ (x4)) and (x6)) ^ (x5)) or ((x1) ^ (x3))) ^ (x6)) and ((x1) ^ (x3)))) ^ (not ((x6) ^ (x2)))) ^ (x1)) or ((x5) ^ (x4))
    
def pf_v7_c20(x0, x1, x2, x3, x4, x5, x6):
    return (((not (((((((not ((((((not ((((x0) ^ (x0)) ^ (x5)) ^ (x0))) ^ (x2)) or (x3)) or (((x6) ^ (x2)) ^ (x6))) and ((x1) or (x5))) or (x3))) ^ (x4)) and (x6)) ^ (x5)) or ((x1) ^ (x3))) ^ (x6)) and ((x1) ^ (x3)))) ^ (not ((x6) ^ (x2)))) ^ (x1)) or ((x5) ^ (x4))
    

@classical_function
def cf_v7_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1) -> Int1:
    return ((not (not ((((not (not (((((((((((((x3) ^ (x6)) or (x1)) ^ (x5)) ^ (x5)) and ((x0) or (x6))) ^ (x5)) ^ (x1)) or (x1)) or (x4)) or (x2)) and (x6)) or ((x2) and ((x2) ^ (x1)))))) or ((x2) ^ (x1))) ^ ((((x0) or (x6)) or (x5)) and (x4))) and (x6)))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))
    
def pf_v7_c21(x0, x1, x2, x3, x4, x5, x6):
    return ((not (not ((((not (not (((((((((((((x3) ^ (x6)) or (x1)) ^ (x5)) ^ (x5)) and ((x0) or (x6))) ^ (x5)) ^ (x1)) or (x1)) or (x4)) or (x2)) and (x6)) or ((x2) and ((x2) ^ (x1)))))) or ((x2) ^ (x1))) ^ ((((x0) or (x6)) or (x5)) and (x4))) and (x6)))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))
    

@classical_function
def cf_v8_c8(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (((not (((((x2) or (x0)) or (x2)) and (x2)) and ((x6) and (x7)))) ^ (x4)) ^ ((x4) and ((x3) or (x5)))) or ((x1) or (x3))
    
def pf_v8_c8(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((not (((((x2) or (x0)) or (x2)) and (x2)) and ((x6) and (x7)))) ^ (x4)) ^ ((x4) and ((x3) or (x5)))) or ((x1) or (x3))
    

@classical_function
def cf_v8_c9(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return ((((((not (not ((x7) or (x6)))) and (x5)) ^ (x5)) or (x2)) ^ (x4)) ^ ((x1) ^ ((x6) or (x0)))) ^ (x3)
    
def pf_v8_c9(x0, x1, x2, x3, x4, x5, x6, x7):
    return ((((((not (not ((x7) or (x6)))) and (x5)) ^ (x5)) or (x2)) ^ (x4)) ^ ((x1) ^ ((x6) or (x0)))) ^ (x3)
    

@classical_function
def cf_v8_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return ((not (not (((((((x5) or (x3)) or (x2)) ^ (x5)) and (x1)) or ((x5) ^ (x6))) or (((x6) and (x5)) or (x0))))) or ((x7) or (x6))) ^ (not (((x6) or (x4)) or (x4)))
    
def pf_v8_c10(x0, x1, x2, x3, x4, x5, x6, x7):
    return ((not (not (((((((x5) or (x3)) or (x2)) ^ (x5)) and (x1)) or ((x5) ^ (x6))) or (((x6) and (x5)) or (x0))))) or ((x7) or (x6))) ^ (not (((x6) or (x4)) or (x4)))
    

@classical_function
def cf_v8_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return not (((((((((((x6) ^ (x4)) or (x2)) and (x7)) and (x5)) and ((x5) and (x7))) or ((x3) ^ (x7))) ^ ((x3) ^ (x7))) ^ ((x3) ^ (x7))) and ((x5) and (x7))) ^ ((((x1) and ((x0) or (x1))) and ((x0) or (x1))) or (x6)))
    
def pf_v8_c11(x0, x1, x2, x3, x4, x5, x6, x7):
    return not (((((((((((x6) ^ (x4)) or (x2)) and (x7)) and (x5)) and ((x5) and (x7))) or ((x3) ^ (x7))) ^ ((x3) ^ (x7))) ^ ((x3) ^ (x7))) and ((x5) and (x7))) ^ ((((x1) and ((x0) or (x1))) and ((x0) or (x1))) or (x6)))
    

@classical_function
def cf_v8_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (((not ((not (not (((((not (x0)) ^ (not (x3))) or (x5)) and (x4)) and (x0)))) and (x6))) and ((x2) ^ (x1))) or ((x2) ^ (x1))) and ((x1) and (x7))
    
def pf_v8_c12(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((not ((not (not (((((not (x0)) ^ (not (x3))) or (x5)) and (x4)) and (x0)))) and (x6))) and ((x2) ^ (x1))) or ((x2) ^ (x1))) and ((x1) and (x7))
    

@classical_function
def cf_v8_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (not ((((((((((not ((x1) ^ (x7))) and (x3)) ^ (x7)) and ((x5) ^ (x4))) or (x6)) or (x0)) and ((x5) or (x5))) ^ (x2)) or (x4)) and (x5))) or (x1)
    
def pf_v8_c13(x0, x1, x2, x3, x4, x5, x6, x7):
    return (not ((((((((((not ((x1) ^ (x7))) and (x3)) ^ (x7)) and ((x5) ^ (x4))) or (x6)) or (x0)) and ((x5) or (x5))) ^ (x2)) or (x4)) and (x5))) or (x1)
    

@classical_function
def cf_v8_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return not ((((((((((((((x2) ^ (x6)) ^ ((x6) ^ (x2))) and (x3)) or (x2)) and ((x3) ^ (x0))) ^ (x0)) ^ (x3)) and (x1)) and ((x6) ^ (x2))) ^ (x0)) ^ ((x0) ^ (((x3) ^ (x0)) or (x5)))) or (x7)) ^ (x4))
    
def pf_v8_c14(x0, x1, x2, x3, x4, x5, x6, x7):
    return not ((((((((((((((x2) ^ (x6)) ^ ((x6) ^ (x2))) and (x3)) or (x2)) and ((x3) ^ (x0))) ^ (x0)) ^ (x3)) and (x1)) and ((x6) ^ (x2))) ^ (x0)) ^ ((x0) ^ (((x3) ^ (x0)) or (x5)))) or (x7)) ^ (x4))
    

@classical_function
def cf_v8_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return ((((((not ((not (((((not ((x3) and (x6))) and (x3)) ^ (x1)) and (x4)) or (x5))) or (x7))) or ((x3) or ((x4) and (x5)))) ^ ((x5) ^ (not (x5)))) and (x0)) ^ (x6)) or ((x2) and (x5))) or ((not (not (x4))) ^ ((x2) and (x6)))
    
def pf_v8_c15(x0, x1, x2, x3, x4, x5, x6, x7):
    return ((((((not ((not (((((not ((x3) and (x6))) and (x3)) ^ (x1)) and (x4)) or (x5))) or (x7))) or ((x3) or ((x4) and (x5)))) ^ ((x5) ^ (not (x5)))) and (x0)) ^ (x6)) or ((x2) and (x5))) or ((not (not (x4))) ^ ((x2) and (x6)))
    

@classical_function
def cf_v8_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return not (((not (not ((((((not ((((((x4) and (x3)) or (x0)) ^ (x0)) ^ (x5)) and (x4))) or ((x6) or (x3))) ^ ((x6) or (x3))) and (x2)) ^ (x3)) and ((x6) and (x1))))) or ((x1) ^ (x3))) ^ (not (x7)))
    
def pf_v8_c16(x0, x1, x2, x3, x4, x5, x6, x7):
    return not (((not (not ((((((not ((((((x4) and (x3)) or (x0)) ^ (x0)) ^ (x5)) and (x4))) or ((x6) or (x3))) ^ ((x6) or (x3))) and (x2)) ^ (x3)) and ((x6) and (x1))))) or ((x1) ^ (x3))) ^ (not (x7)))
    

@classical_function
def cf_v8_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return ((((not ((((not (((not ((((((x3) or (x3)) ^ (x3)) or (x7)) ^ (x0)) or (x1))) ^ (not (x0))) or ((x4) or ((x4) or (x0))))) ^ ((x2) ^ (x2))) and ((x5) or ((x4) or ((x4) or (x0))))) ^ ((x7) ^ (not (x0))))) or (x4)) or (not (((x5) or (x4)) ^ (x6)))) ^ (x0)) ^ (x7)
    
def pf_v8_c17(x0, x1, x2, x3, x4, x5, x6, x7):
    return ((((not ((((not (((not ((((((x3) or (x3)) ^ (x3)) or (x7)) ^ (x0)) or (x1))) ^ (not (x0))) or ((x4) or ((x4) or (x0))))) ^ ((x2) ^ (x2))) and ((x5) or ((x4) or ((x4) or (x0))))) ^ ((x7) ^ (not (x0))))) or (x4)) or (not (((x5) or (x4)) ^ (x6)))) ^ (x0)) ^ (x7)
    

@classical_function
def cf_v8_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (((((not ((((not ((not ((not (((((x7) ^ (x0)) or (x6)) or ((x7) ^ (x3))) ^ (x1))) ^ (x1))) ^ ((x2) ^ (x0)))) and ((x2) ^ (x0))) or (x2)) or (x2))) and (x5)) and (not ((x3) ^ (x1)))) or (not (x4))) ^ (x1)) and ((x1) or (x5))
    
def pf_v8_c18(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((((not ((((not ((not ((not (((((x7) ^ (x0)) or (x6)) or ((x7) ^ (x3))) ^ (x1))) ^ (x1))) ^ ((x2) ^ (x0)))) and ((x2) ^ (x0))) or (x2)) or (x2))) and (x5)) and (not ((x3) ^ (x1)))) or (not (x4))) ^ (x1)) and ((x1) or (x5))
    

@classical_function
def cf_v8_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (((not ((not (((((not ((((not (((((x0) or (x2)) ^ (x0)) or (x4)) or (x4))) or (x5)) ^ ((x3) or ((x3) or ((x3) or (x6))))) or ((x1) or (x3)))) ^ (x1)) ^ (((x1) or (x3)) ^ (x6))) and (x7)) or (x7))) or ((x3) or ((x3) or (x6))))) and ((x3) or ((x3) or ((x3) or (x6))))) and ((x6) and (x4))) and (x1)
    
def pf_v8_c19(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((not ((not (((((not ((((not (((((x0) or (x2)) ^ (x0)) or (x4)) or (x4))) or (x5)) ^ ((x3) or ((x3) or ((x3) or (x6))))) or ((x1) or (x3)))) ^ (x1)) ^ (((x1) or (x3)) ^ (x6))) and (x7)) or (x7))) or ((x3) or ((x3) or (x6))))) and ((x3) or ((x3) or ((x3) or (x6))))) and ((x6) and (x4))) and (x1)
    

@classical_function
def cf_v8_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return ((((((((((not (((not ((((not (((x2) and (x2)) and (x5))) or (x6)) and (x7)) and (x5))) and (x2)) ^ ((x1) and (x3)))) and (x3)) or (x5)) and (x7)) and ((x1) and (x4))) and (x3)) and ((x3) and ((x0) or (not (x2))))) ^ (x4)) and (((x1) and (x3)) or (((x0) or (not (x2))) and (x2)))) and (x7)) ^ ((x0) or (not (x2)))
    
def pf_v8_c20(x0, x1, x2, x3, x4, x5, x6, x7):
    return ((((((((((not (((not ((((not (((x2) and (x2)) and (x5))) or (x6)) and (x7)) and (x5))) and (x2)) ^ ((x1) and (x3)))) and (x3)) or (x5)) and (x7)) and ((x1) and (x4))) and (x3)) and ((x3) and ((x0) or (not (x2))))) ^ (x4)) and (((x1) and (x3)) or (((x0) or (not (x2))) and (x2)))) and (x7)) ^ ((x0) or (not (x2)))
    

@classical_function
def cf_v8_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (not ((((not ((((((not ((not ((not (((not (((x6) ^ (x3)) ^ (x4))) ^ (x2)) and (x2))) and ((x1) ^ (x7)))) and (x4))) and (not (x0))) or (x0)) and ((x5) ^ (x7))) ^ ((not (x7)) and (x2))) ^ (x2))) and (not (x1))) ^ (((x5) ^ (x7)) and (((x3) and (x2)) or (x6)))) ^ ((x3) and (x2)))) and ((not (x5)) and (x7))
    
def pf_v8_c21(x0, x1, x2, x3, x4, x5, x6, x7):
    return (not ((((not ((((((not ((not ((not (((not (((x6) ^ (x3)) ^ (x4))) ^ (x2)) and (x2))) and ((x1) ^ (x7)))) and (x4))) and (not (x0))) or (x0)) and ((x5) ^ (x7))) ^ ((not (x7)) and (x2))) ^ (x2))) and (not (x1))) ^ (((x5) ^ (x7)) and (((x3) and (x2)) or (x6)))) ^ ((x3) and (x2)))) and ((not (x5)) and (x7))
    

@classical_function
def cf_v8_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1) -> Int1:
    return (((not ((((not ((((((((not (not ((((not ((x7) or (x0))) ^ (x4)) and ((x4) or (x0))) or (x5)))) ^ ((x5) or (x1))) or (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) or (x5)) ^ (x2)) and (((x4) or (x0)) ^ (x6))) or (x6)) and (x1))) and (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) ^ (not (x7))) ^ (((x1) and (x7)) ^ (x4)))) or ((not (x0)) ^ (not ((x4) or (x3))))) and (((not (((x4) or (x0)) ^ (x6))) or (x0)) ^ ((x5) or (x1)))) ^ ((x1) or (x2))
    
def pf_v8_c22(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((not ((((not ((((((((not (not ((((not ((x7) or (x0))) ^ (x4)) and ((x4) or (x0))) or (x5)))) ^ ((x5) or (x1))) or (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) or (x5)) ^ (x2)) and (((x4) or (x0)) ^ (x6))) or (x6)) and (x1))) and (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) ^ (not (x7))) ^ (((x1) and (x7)) ^ (x4)))) or ((not (x0)) ^ (not ((x4) or (x3))))) and (((not (((x4) or (x0)) ^ (x6))) or (x0)) ^ ((x5) or (x1)))) ^ ((x1) or (x2))
    

@classical_function
def cf_v9_c10(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((((((((((x5) or (x4)) and (x3)) and (x4)) or (x7)) or ((x8) and (x8))) or (x8)) ^ ((x8) or (x5))) and (x2)) ^ ((x8) and (x8))) and ((x6) ^ ((x0) and (x1)))
    
def pf_v9_c10(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((((((((((x5) or (x4)) and (x3)) and (x4)) or (x7)) or ((x8) and (x8))) or (x8)) ^ ((x8) or (x5))) and (x2)) ^ ((x8) and (x8))) and ((x6) ^ ((x0) and (x1)))
    

@classical_function
def cf_v9_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (not ((((((((not ((x2) or (x1))) and (not ((x6) or (x5)))) ^ (not ((x6) or (x5)))) and (x3)) and (x0)) ^ (x5)) ^ (x4)) or (((x0) and (x7)) and ((x4) ^ (x8))))) and (((x0) and (x7)) ^ (x8))
    
def pf_v9_c11(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (not ((((((((not ((x2) or (x1))) and (not ((x6) or (x5)))) ^ (not ((x6) or (x5)))) and (x3)) and (x0)) ^ (x5)) ^ (x4)) or (((x0) and (x7)) and ((x4) ^ (x8))))) and (((x0) and (x7)) ^ (x8))
    

@classical_function
def cf_v9_c12(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((not ((((((((((x8) or (x8)) and (x6)) and (x0)) and (x8)) ^ (x4)) and ((x8) and (x8))) or (x4)) ^ (x3)) or (x7))) or ((x5) ^ (((x1) or (x6)) and (x1)))) and (x2)
    
def pf_v9_c12(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((not ((((((((((x8) or (x8)) and (x6)) and (x0)) and (x8)) ^ (x4)) and ((x8) and (x8))) or (x4)) ^ (x3)) or (x7))) or ((x5) ^ (((x1) or (x6)) and (x1)))) and (x2)
    

@classical_function
def cf_v9_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (not ((not (((not ((((not (((x0) or (x2)) or (x3))) and (x5)) and (x0)) or (x6))) and ((x7) and (x3))) and (((x8) or (x8)) and ((x3) or (x3))))) or ((x6) and (x1)))) and ((x8) or ((x4) and (x2)))
    
def pf_v9_c13(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (not ((not (((not ((((not (((x0) or (x2)) or (x3))) and (x5)) and (x0)) or (x6))) and ((x7) and (x3))) and (((x8) or (x8)) and ((x3) or (x3))))) or ((x6) and (x1)))) and ((x8) or ((x4) and (x2)))
    

@classical_function
def cf_v9_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (((((((((((((not (x4)) and (x5)) ^ (x8)) or (x3)) and ((x7) or (x7))) and (x5)) and (x8)) or (x6)) or (x4)) or (x4)) or ((x7) or (x7))) or ((x7) or (x7))) ^ (((x2) ^ (x1)) or (x0))) or ((x6) and (((x7) or (x7)) and (x2)))
    
def pf_v9_c14(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (((((((((((((not (x4)) and (x5)) ^ (x8)) or (x3)) and ((x7) or (x7))) and (x5)) and (x8)) or (x6)) or (x4)) or (x4)) or ((x7) or (x7))) or ((x7) or (x7))) ^ (((x2) ^ (x1)) or (x0))) or ((x6) and (((x7) or (x7)) and (x2)))
    

@classical_function
def cf_v9_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (((not ((((((not ((not ((((x6) ^ (x5)) or (x2)) and (x6))) or (not (x3)))) ^ (x2)) and ((x2) or (x4))) or ((x0) and (x4))) or ((x1) or ((x0) or (x2)))) ^ (((x8) and ((x0) and (x4))) or (not (x3))))) ^ (x7)) ^ (x2)) and (x8)
    
def pf_v9_c15(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (((not ((((((not ((not ((((x6) ^ (x5)) or (x2)) and (x6))) or (not (x3)))) ^ (x2)) and ((x2) or (x4))) or ((x0) and (x4))) or ((x1) or ((x0) or (x2)))) ^ (((x8) and ((x0) and (x4))) or (not (x3))))) ^ (x7)) ^ (x2)) and (x8)
    

@classical_function
def cf_v9_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((((((((((not (not ((not (((x5) ^ (x4)) and (x7))) or ((x6) ^ (x6))))) and (x2)) or (not (x3))) or (((x5) and (x8)) or (not (x3)))) or ((x0) or (not (x8)))) or (x2)) and (not ((x5) and (x8)))) or ((x0) or (not (x8)))) ^ ((not (x8)) or (not (x8)))) and (x2)) and ((x1) or (not (x3)))
    
def pf_v9_c16(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((((((((((not (not ((not (((x5) ^ (x4)) and (x7))) or ((x6) ^ (x6))))) and (x2)) or (not (x3))) or (((x5) and (x8)) or (not (x3)))) or ((x0) or (not (x8)))) or (x2)) and (not ((x5) and (x8)))) or ((x0) or (not (x8)))) ^ ((not (x8)) or (not (x8)))) and (x2)) and ((x1) or (not (x3)))
    

@classical_function
def cf_v9_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (not (not (((((((not ((((((not ((x2) and (x0))) or (x2)) and (x6)) ^ ((x5) and (x5))) and ((x4) or (x8))) and (not (x3)))) or (x7)) and (not (x3))) or (x1)) ^ (x2)) or ((x5) and (x5))) or ((x4) or (x8))))) or ((x1) or (x2))
    
def pf_v9_c17(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (not (not (((((((not ((((((not ((x2) and (x0))) or (x2)) and (x6)) ^ ((x5) and (x5))) and ((x4) or (x8))) and (not (x3)))) or (x7)) and (not (x3))) or (x1)) ^ (x2)) or ((x5) and (x5))) or ((x4) or (x8))))) or ((x1) or (x2))
    

@classical_function
def cf_v9_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((((not ((((((not ((not (not (not ((((x6) or (x5)) or (x4)) or (x7))))) and (x8))) and (x3)) or (x6)) or (not (x2))) and (not (x1))) or (not (x5)))) and ((x6) or (((x1) and (x4)) and (x0)))) ^ (x6)) and ((x4) or ((x4) or (x7)))) ^ (not (x2))
    
def pf_v9_c18(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((((not ((((((not ((not (not (not ((((x6) or (x5)) or (x4)) or (x7))))) and (x8))) and (x3)) or (x6)) or (not (x2))) and (not (x1))) or (not (x5)))) and ((x6) or (((x1) and (x4)) and (x0)))) ^ (x6)) and ((x4) or ((x4) or (x7)))) ^ (not (x2))
    

@classical_function
def cf_v9_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (((((((((((((not (not (((((x8) or (x1)) or (x1)) or (x2)) and (x7)))) and (not (x3))) and (not (x0))) ^ (x5)) and (x4)) ^ (x5)) ^ ((x7) or (x2))) and (x7)) and (((x4) and (x7)) and ((x4) and (x7)))) or (((x1) or (x6)) and ((x1) or (x6)))) and ((((x1) or (x6)) and (x2)) or (((x1) or (x6)) and ((x1) or (x6))))) ^ (x3)) ^ (x6)) ^ (not (x3))
    
def pf_v9_c19(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (((((((((((((not (not (((((x8) or (x1)) or (x1)) or (x2)) and (x7)))) and (not (x3))) and (not (x0))) ^ (x5)) and (x4)) ^ (x5)) ^ ((x7) or (x2))) and (x7)) and (((x4) and (x7)) and ((x4) and (x7)))) or (((x1) or (x6)) and ((x1) or (x6)))) and ((((x1) or (x6)) and (x2)) or (((x1) or (x6)) and ((x1) or (x6))))) ^ (x3)) ^ (x6)) ^ (not (x3))
    

@classical_function
def cf_v9_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((not (((not (((((not ((((((((((x0) and (x6)) or ((x0) and (x6))) or (x5)) or ((x0) and (x6))) ^ (x3)) ^ (x0)) or (x2)) or ((x5) or (x8))) and ((((x2) or (x4)) ^ (x4)) or (x4)))) and ((((x2) or (x4)) ^ (x4)) or (x4))) or (x7)) ^ (x2)) ^ (((x2) or (x4)) or ((x0) and (x6))))) and (x8)) or ((((x2) or (x4)) ^ (x4)) or (x4)))) ^ (x3)) and ((x4) and (((((x2) or (x4)) ^ (x4)) or (x4)) and (x1)))
    
def pf_v9_c20(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((not (((not (((((not ((((((((((x0) and (x6)) or ((x0) and (x6))) or (x5)) or ((x0) and (x6))) ^ (x3)) ^ (x0)) or (x2)) or ((x5) or (x8))) and ((((x2) or (x4)) ^ (x4)) or (x4)))) and ((((x2) or (x4)) ^ (x4)) or (x4))) or (x7)) ^ (x2)) ^ (((x2) or (x4)) or ((x0) and (x6))))) and (x8)) or ((((x2) or (x4)) ^ (x4)) or (x4)))) ^ (x3)) and ((x4) and (((((x2) or (x4)) ^ (x4)) or (x4)) and (x1)))
    

@classical_function
def cf_v9_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((((((not ((not (((((not (((((not (((x1) and (x6)) ^ (x5))) and ((x3) and (x1))) and (x3)) ^ (x0)) and (x7))) ^ (x4)) or (x1)) or (x7)) ^ (not (x3)))) ^ ((x2) and (x5)))) and (x6)) ^ (x0)) or ((((x3) and (x1)) ^ ((x3) and (x1))) and ((x2) and (x5)))) or (x2)) ^ (x3)) or ((x8) and ((x3) and (x1)))
    
def pf_v9_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((((((not ((not (((((not (((((not (((x1) and (x6)) ^ (x5))) and ((x3) and (x1))) and (x3)) ^ (x0)) and (x7))) ^ (x4)) or (x1)) or (x7)) ^ (not (x3)))) ^ ((x2) and (x5)))) and (x6)) ^ (x0)) or ((((x3) and (x1)) ^ ((x3) and (x1))) and ((x2) and (x5)))) or (x2)) ^ (x3)) or ((x8) and ((x3) and (x1)))
    

@classical_function
def cf_v9_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return ((((((((((not (not ((not (((not ((((((x2) or (x7)) and (x7)) or (x2)) or (x3)) and ((x4) or ((x6) and (x1))))) or (not (x6))) or (x0))) and (x7)))) or (not (not (x6)))) or ((x6) and (x1))) ^ (x2)) or (x5)) or (not (x4))) and (not (not (x6)))) and (x4)) and (((x3) and (x1)) ^ ((x7) or (x0)))) and ((x5) and (x7))) and ((((x8) and (x3)) or (x6)) or ((((x8) and (x3)) or (x6)) or ((x6) and (x1))))
    
def pf_v9_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return ((((((((((not (not ((not (((not ((((((x2) or (x7)) and (x7)) or (x2)) or (x3)) and ((x4) or ((x6) and (x1))))) or (not (x6))) or (x0))) and (x7)))) or (not (not (x6)))) or ((x6) and (x1))) ^ (x2)) or (x5)) or (not (x4))) and (not (not (x6)))) and (x4)) and (((x3) and (x1)) ^ ((x7) or (x0)))) and ((x5) and (x7))) and ((((x8) and (x3)) or (x6)) or ((((x8) and (x3)) or (x6)) or ((x6) and (x1))))
    

@classical_function
def cf_v9_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1) -> Int1:
    return (not (((((((not (not ((not (not ((((((((not (((x6) or (x1)) ^ (x7))) or (x4)) and (x8)) and ((x0) and (x7))) and (x5)) and ((x0) and (x7))) ^ (x3)) ^ ((x6) and ((x2) or (x0)))))) ^ ((x2) ^ (x5))))) or (x1)) or (((x2) ^ (x5)) and (x1))) and (((x2) or (x0)) or (x0))) ^ ((x1) ^ (x4))) or (not (not (x8)))) and (not (x1)))) ^ (((x2) or (x0)) or (x0))
    
def pf_v9_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return (not (((((((not (not ((not (not ((((((((not (((x6) or (x1)) ^ (x7))) or (x4)) and (x8)) and ((x0) and (x7))) and (x5)) and ((x0) and (x7))) ^ (x3)) ^ ((x6) and ((x2) or (x0)))))) ^ ((x2) ^ (x5))))) or (x1)) or (((x2) ^ (x5)) and (x1))) and (((x2) or (x0)) or (x0))) ^ ((x1) ^ (x4))) or (not (not (x8)))) and (not (x1)))) ^ (((x2) or (x0)) or (x0))
    

@classical_function
def cf_v10_c13(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((((not (((((((((x0) ^ (x4)) ^ (x5)) and ((x8) ^ (x2))) and ((x3) ^ ((x8) ^ (x2)))) and (x3)) and ((x3) ^ ((x8) ^ (x2)))) or (x7)) ^ (x4))) or ((x3) ^ ((x8) ^ (x2)))) ^ (not (x1))) ^ ((x9) and (x6))) ^ (x1)
    
def pf_v10_c13(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((((not (((((((((x0) ^ (x4)) ^ (x5)) and ((x8) ^ (x2))) and ((x3) ^ ((x8) ^ (x2)))) and (x3)) and ((x3) ^ ((x8) ^ (x2)))) or (x7)) ^ (x4))) or ((x3) ^ ((x8) ^ (x2)))) ^ (not (x1))) ^ ((x9) and (x6))) ^ (x1)
    

@classical_function
def cf_v10_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((((not (((((not ((((((x1) ^ (x0)) ^ (x8)) ^ (x4)) ^ (x4)) ^ (x1))) ^ (x2)) ^ (x3)) and (x9)) or (x7))) and (x3)) or (((x4) ^ (x6)) or (x9))) ^ ((x4) ^ (x9))) ^ (((x5) and ((x4) ^ (x6))) and (x5))
    
def pf_v10_c15(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((((not (((((not ((((((x1) ^ (x0)) ^ (x8)) ^ (x4)) ^ (x4)) ^ (x1))) ^ (x2)) ^ (x3)) and (x9)) or (x7))) and (x3)) or (((x4) ^ (x6)) or (x9))) ^ ((x4) ^ (x9))) ^ (((x5) and ((x4) ^ (x6))) and (x5))
    

@classical_function
def cf_v10_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((not (((((((((not ((((not (x6)) or (not (x4))) ^ (x6)) ^ (x1))) or (x0)) and (x1)) or (not ((x3) or (x9)))) and (x2)) or (x7)) or (x5)) ^ (x5)) or (not ((x3) or (x9))))) or (x8)) ^ ((x2) or (not ((x3) or (x9))))
    
def pf_v10_c16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((not (((((((((not ((((not (x6)) or (not (x4))) ^ (x6)) ^ (x1))) or (x0)) and (x1)) or (not ((x3) or (x9)))) and (x2)) or (x7)) or (x5)) ^ (x5)) or (not ((x3) or (x9))))) or (x8)) ^ ((x2) or (not ((x3) or (x9))))
    

@classical_function
def cf_v10_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((not (((((not ((((((((((x4) and (x9)) ^ (x1)) or (x9)) or ((x3) or (x2))) or (x5)) ^ (x7)) and (x2)) and ((x5) or (x5))) or ((x8) and (x5)))) and (x7)) and (x9)) or ((x6) ^ ((x3) and (x2)))) ^ (x1))) and (x5)) or (((x0) and (x8)) or (((x6) ^ ((x3) and (x2))) ^ ((x1) or (x5))))
    
def pf_v10_c17(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((not (((((not ((((((((((x4) and (x9)) ^ (x1)) or (x9)) or ((x3) or (x2))) or (x5)) ^ (x7)) and (x2)) and ((x5) or (x5))) or ((x8) and (x5)))) and (x7)) and (x9)) or ((x6) ^ ((x3) and (x2)))) ^ (x1))) and (x5)) or (((x0) and (x8)) or (((x6) ^ ((x3) and (x2))) ^ ((x1) or (x5))))
    

@classical_function
def cf_v10_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return not (((((((((not (not ((((((((x2) or (x5)) ^ (x9)) or (x8)) ^ (x6)) ^ (x4)) or ((x6) ^ ((x2) ^ (x6)))) or (x7)))) and (x1)) and ((x6) ^ (x3))) ^ (((x4) or (x6)) and (not ((x6) ^ ((x2) ^ (x6)))))) or (x0)) ^ (x3)) or (x1)) ^ (x3)) and ((x2) or (x0)))
    
def pf_v10_c18(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return not (((((((((not (not ((((((((x2) or (x5)) ^ (x9)) or (x8)) ^ (x6)) ^ (x4)) or ((x6) ^ ((x2) ^ (x6)))) or (x7)))) and (x1)) and ((x6) ^ (x3))) ^ (((x4) or (x6)) and (not ((x6) ^ ((x2) ^ (x6)))))) or (x0)) ^ (x3)) or (x1)) ^ (x3)) and ((x2) or (x0)))
    

@classical_function
def cf_v10_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return not (not ((not ((((((((((((((((x9) or (x5)) and (x4)) ^ (x5)) or (x4)) and (x1)) and (x7)) and (x2)) ^ ((x9) ^ (x1))) or (x0)) and ((x8) or (x3))) or (x4)) ^ ((x1) and (x2))) ^ ((x9) ^ (x1))) ^ ((x4) and ((x8) or (x6)))) and (((x8) or (x3)) and (x7)))) and (x5)))
    
def pf_v10_c19(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return not (not ((not ((((((((((((((((x9) or (x5)) and (x4)) ^ (x5)) or (x4)) and (x1)) and (x7)) and (x2)) ^ ((x9) ^ (x1))) or (x0)) and ((x8) or (x3))) or (x4)) ^ ((x1) and (x2))) ^ ((x9) ^ (x1))) ^ ((x4) and ((x8) or (x6)))) and (((x8) or (x3)) and (x7)))) and (x5)))
    

@classical_function
def cf_v10_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return (not ((not (not (not (not (((((not (((not (not ((not (((x2) and (x5)) and (x6))) ^ (x4)))) ^ (x8)) ^ (x2))) ^ ((x0) and (x1))) or ((x9) ^ (x3))) or (x8)) and (not (x4))))))) ^ ((x6) and (((x0) and (x1)) ^ (x7))))) or ((x9) ^ (x3))
    
def pf_v10_c20(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (not ((not (not (not (not (((((not (((not (not ((not (((x2) and (x5)) and (x6))) ^ (x4)))) ^ (x8)) ^ (x2))) ^ ((x0) and (x1))) or ((x9) ^ (x3))) or (x8)) and (not (x4))))))) ^ ((x6) and (((x0) and (x1)) ^ (x7))))) or ((x9) ^ (x3))
    

@classical_function
def cf_v10_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return (((((not ((((not ((((((((((((x8) and (x5)) and (x6)) and (x2)) or (x1)) and ((x6) and (x3))) and (x5)) and (x9)) and (((x6) and (x3)) and (x5))) or (((x9) and (x8)) and (x1))) and ((x6) and (x3))) and (x4))) ^ ((x6) or (x7))) ^ (((x9) and (x8)) and (x1))) and (x1))) and ((x1) or (x4))) ^ (x3)) ^ (x7)) ^ ((x9) ^ (x0))) and (x3)
    
def pf_v10_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (((((not ((((not ((((((((((((x8) and (x5)) and (x6)) and (x2)) or (x1)) and ((x6) and (x3))) and (x5)) and (x9)) and (((x6) and (x3)) and (x5))) or (((x9) and (x8)) and (x1))) and ((x6) and (x3))) and (x4))) ^ ((x6) or (x7))) ^ (((x9) and (x8)) and (x1))) and (x1))) and ((x1) or (x4))) ^ (x3)) ^ (x7)) ^ ((x9) ^ (x0))) and (x3)
    

@classical_function
def cf_v10_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return (((((((not ((not (((((((((((((x1) ^ (x3)) and (x2)) ^ (x2)) ^ (x1)) and (x4)) or ((x9) and ((x7) ^ (x9)))) and (x2)) ^ ((x3) ^ (x6))) or (x2)) or (x9)) or ((x9) and ((x7) ^ (x9)))) and (((x0) or (x4)) or (x9)))) ^ ((x7) ^ (x9)))) or (x6)) or (x8)) ^ (x4)) and (((x7) ^ (x9)) ^ (x5))) ^ (((x3) and (x4)) or (x9))) or (((x0) or (x4)) or (x9))) and ((x9) ^ (x4))
    
def pf_v10_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (((((((not ((not (((((((((((((x1) ^ (x3)) and (x2)) ^ (x2)) ^ (x1)) and (x4)) or ((x9) and ((x7) ^ (x9)))) and (x2)) ^ ((x3) ^ (x6))) or (x2)) or (x9)) or ((x9) and ((x7) ^ (x9)))) and (((x0) or (x4)) or (x9)))) ^ ((x7) ^ (x9)))) or (x6)) or (x8)) ^ (x4)) and (((x7) ^ (x9)) ^ (x5))) ^ (((x3) and (x4)) or (x9))) or (((x0) or (x4)) or (x9))) and ((x9) ^ (x4))
    

@classical_function
def cf_v10_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((((((not (((((((not (not (not (((((not (((x0) and (x2)) ^ ((x6) and (x0)))) or (x5)) and ((x8) ^ (x1))) ^ (x8)) ^ (x4))))) ^ ((x9) or (x9))) or (not (x0))) ^ ((x7) ^ (x6))) and (((x6) ^ (x8)) ^ (x9))) and (((x8) ^ (x1)) and (x9))) and (x9))) or (x3)) ^ (x9)) ^ (((x9) and (x4)) ^ (x5))) and (x7)) or (not (x4))) and ((x6) ^ (x8))
    
def pf_v10_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((((((not (((((((not (not (not (((((not (((x0) and (x2)) ^ ((x6) and (x0)))) or (x5)) and ((x8) ^ (x1))) ^ (x8)) ^ (x4))))) ^ ((x9) or (x9))) or (not (x0))) ^ ((x7) ^ (x6))) and (((x6) ^ (x8)) ^ (x9))) and (((x8) ^ (x1)) and (x9))) and (x9))) or (x3)) ^ (x9)) ^ (((x9) and (x4)) ^ (x5))) and (x7)) or (not (x4))) and ((x6) ^ (x8))
    

@classical_function
def cf_v10_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1) -> Int1:
    return ((((not (not (((not (((not (((((((((((not ((x5) ^ (x5))) and (not (x1))) and (x8)) and ((x6) ^ (x7))) or ((x5) and (x9))) or (x2)) or (x3)) and (x5)) and ((x6) ^ (x7))) and ((x2) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) or (x9))) or (x9)) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) ^ (((x6) ^ (x7)) and ((x6) ^ (x7)))) ^ (x1)))) or ((x8) ^ ((x1) or (x0)))) or (not ((not (x1)) ^ (x5)))) ^ (x4)) or (x6)
    
def pf_v10_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((((not (not (((not (((not (((((((((((not ((x5) ^ (x5))) and (not (x1))) and (x8)) and ((x6) ^ (x7))) or ((x5) and (x9))) or (x2)) or (x3)) and (x5)) and ((x6) ^ (x7))) and ((x2) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) or (x9))) or (x9)) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) ^ (((x6) ^ (x7)) and ((x6) ^ (x7)))) ^ (x1)))) or ((x8) ^ ((x1) or (x0)))) or (not ((not (x1)) ^ (x5)))) ^ (x4)) or (x6)
    

@classical_function
def cf_v11_c11(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (not (((((not (((((x0) and (x0)) and (x10)) and (x6)) and (x2))) and (x4)) ^ (x5)) ^ (((x6) and (not ((x2) and (x10)))) ^ (not ((x2) and (x10))))) and (((x3) and (x7)) and (x9)))) and ((x8) ^ (x10))
    
def pf_v11_c11(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (not (((((not (((((x0) and (x0)) and (x10)) and (x6)) and (x2))) and (x4)) ^ (x5)) ^ (((x6) and (not ((x2) and (x10)))) ^ (not ((x2) and (x10))))) and (((x3) and (x7)) and (x9)))) and ((x8) ^ (x10))
    

@classical_function
def cf_v11_c14(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (not ((((((((not (((((x4) and (x9)) ^ (x3)) ^ (x8)) ^ ((x5) ^ (x3)))) or (not (x10))) or (x3)) or (x6)) ^ (x2)) ^ (x8)) or ((((x8) ^ (x6)) or (x0)) or (x7))) or (x4))) or (x1)
    
def pf_v11_c14(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (not ((((((((not (((((x4) and (x9)) ^ (x3)) ^ (x8)) ^ ((x5) ^ (x3)))) or (not (x10))) or (x3)) or (x6)) ^ (x2)) ^ (x8)) or ((((x8) ^ (x6)) or (x0)) or (x7))) or (x4))) or (x1)
    

@classical_function
def cf_v11_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (((not ((((not (not ((((not (((x5) and (x0)) ^ (x2))) or (x3)) and (x4)) or (x6)))) ^ (((x3) ^ (x8)) or ((x7) and (x6)))) or (((x3) ^ (x8)) or ((x7) and (x6)))) and (((x3) ^ (x8)) or ((x7) and (x6))))) ^ (x3)) and (x9)) or (x10)
    
def pf_v11_c15(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (((not ((((not (not ((((not (((x5) and (x0)) ^ (x2))) or (x3)) and (x4)) or (x6)))) ^ (((x3) ^ (x8)) or ((x7) and (x6)))) or (((x3) ^ (x8)) or ((x7) and (x6)))) and (((x3) ^ (x8)) or ((x7) and (x6))))) ^ (x3)) and (x9)) or (x10)
    

@classical_function
def cf_v11_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (not (((((not ((not (((((((not (x6)) ^ (x10)) and (x3)) and (x4)) and (x3)) or (x8)) and (x8))) and (x2))) ^ ((x10) and (x5))) ^ (not (x9))) and ((x10) ^ (x3))) or (x7))) or (x0)
    
def pf_v11_c16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (not (((((not ((not (((((((not (x6)) ^ (x10)) and (x3)) and (x4)) and (x3)) or (x8)) and (x8))) and (x2))) ^ ((x10) and (x5))) ^ (not (x9))) and ((x10) ^ (x3))) or (x7))) or (x0)
    

@classical_function
def cf_v11_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (((((not (not (((((not ((not ((((x6) or (x3)) and (x3)) or (x7))) ^ (x9))) and (x5)) or ((x4) ^ (x3))) ^ ((x2) or (x10))) ^ (((x10) and ((x9) and (x2))) ^ (x1))))) or (not ((x6) or (x4)))) or (x3)) ^ (x0)) ^ (((x10) and ((x9) and (x2))) ^ (x1))) or ((x8) or (x5))
    
def pf_v11_c17(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (((((not (not (((((not ((not ((((x6) or (x3)) and (x3)) or (x7))) ^ (x9))) and (x5)) or ((x4) ^ (x3))) ^ ((x2) or (x10))) ^ (((x10) and ((x9) and (x2))) ^ (x1))))) or (not ((x6) or (x4)))) or (x3)) ^ (x0)) ^ (((x10) and ((x9) and (x2))) ^ (x1))) or ((x8) or (x5))
    

@classical_function
def cf_v11_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return ((((not (not (((((not (((not ((not (((x5) and (x5)) ^ (x8))) ^ ((x0) and (x6)))) and (x7)) ^ (x3))) or ((x0) and (x6))) or ((x4) ^ (x4))) ^ (x7)) ^ (not (x4))))) or ((not (x10)) or (x9))) ^ ((x0) and (x6))) or ((x7) and ((x3) ^ (x2)))) ^ ((x3) or (x9))
    
def pf_v11_c18(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return ((((not (not (((((not (((not ((not (((x5) and (x5)) ^ (x8))) ^ ((x0) and (x6)))) and (x7)) ^ (x3))) or ((x0) and (x6))) or ((x4) ^ (x4))) ^ (x7)) ^ (not (x4))))) or ((not (x10)) or (x9))) ^ ((x0) and (x6))) or ((x7) and ((x3) ^ (x2)))) ^ ((x3) or (x9))
    

@classical_function
def cf_v11_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return ((((((((not (not ((not ((((not ((((x5) or (x5)) ^ ((x0) or (x7))) or (x6))) or (x0)) or (x7)) ^ (x10))) ^ (x4)))) ^ ((x2) ^ (x10))) or (x8)) or (x8)) or (not (x0))) ^ (((not (x9)) ^ (x2)) or (not (x10)))) or ((((x0) or (x7)) or (x9)) or (x3))) ^ (not (x9))) and (x6)
    
def pf_v11_c19(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return ((((((((not (not ((not ((((not ((((x5) or (x5)) ^ ((x0) or (x7))) or (x6))) or (x0)) or (x7)) ^ (x10))) ^ (x4)))) ^ ((x2) ^ (x10))) or (x8)) or (x8)) or (not (x0))) ^ (((not (x9)) ^ (x2)) or (not (x10)))) or ((((x0) or (x7)) or (x9)) or (x3))) ^ (not (x9))) and (x6)
    

@classical_function
def cf_v11_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (((not (((((not ((((((((((((x0) and (x3)) and (x5)) and (x4)) ^ (x0)) ^ (x3)) ^ (x5)) and ((x9) and (x10))) or (x10)) and ((x3) and (x6))) or (x10)) and ((x8) ^ (x1)))) and (((x9) and (x10)) and ((x3) ^ (x8)))) and (((x8) ^ (x1)) or (x3))) or (x2)) ^ (x7))) or (((x8) ^ (x1)) or (x3))) ^ ((x4) and ((x3) and (x6)))) and (x0)
    
def pf_v11_c20(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (((not (((((not ((((((((((((x0) and (x3)) and (x5)) and (x4)) ^ (x0)) ^ (x3)) ^ (x5)) and ((x9) and (x10))) or (x10)) and ((x3) and (x6))) or (x10)) and ((x8) ^ (x1)))) and (((x9) and (x10)) and ((x3) ^ (x8)))) and (((x8) ^ (x1)) or (x3))) or (x2)) ^ (x7))) or (((x8) ^ (x1)) or (x3))) ^ ((x4) and ((x3) and (x6)))) and (x0)
    

@classical_function
def cf_v11_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return not (((((not (not (((((((((not ((not (((not (x9)) or (x2)) or (x5))) or (x6))) or (x0)) ^ ((x10) or (x8))) or ((x1) ^ (x3))) or (x1)) and (x3)) ^ (x7)) or ((x7) and (x1))) ^ ((x7) and (x1))))) ^ ((((x1) ^ (x3)) or (x1)) or (x4))) and (not (x8))) and ((x7) and (x1))) and ((x10) ^ (x3)))
    
def pf_v11_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return not (((((not (not (((((((((not ((not (((not (x9)) or (x2)) or (x5))) or (x6))) or (x0)) ^ ((x10) or (x8))) or ((x1) ^ (x3))) or (x1)) and (x3)) ^ (x7)) or ((x7) and (x1))) ^ ((x7) and (x1))))) ^ ((((x1) ^ (x3)) or (x1)) or (x4))) and (not (x8))) and ((x7) and (x1))) and ((x10) ^ (x3)))
    

@classical_function
def cf_v11_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (((not ((not (not (((not (((not ((not ((not ((((((x10) or (x9)) or (x2)) ^ (x6)) or (x1)) ^ (x9))) or (x1))) and (not (x8)))) or ((x9) or ((x5) or ((x8) or (x0))))) ^ (x4))) or (x3)) or (x1)))) or (not ((x5) or ((x8) or (x0)))))) or (x0)) and ((x7) or (x8))) ^ (x9)
    
def pf_v11_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (((not ((not (not (((not (((not ((not ((not ((((((x10) or (x9)) or (x2)) ^ (x6)) or (x1)) ^ (x9))) or (x1))) and (not (x8)))) or ((x9) or ((x5) or ((x8) or (x0))))) ^ (x4))) or (x3)) or (x1)))) or (not ((x5) or ((x8) or (x0)))))) or (x0)) and ((x7) or (x8))) ^ (x9)
    

@classical_function
def cf_v11_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return not (not ((((((not ((not (((((((((((not (((x6) or (x5)) or (x7))) ^ (x7)) or (x8)) and (x3)) and (x7)) and ((x1) or (x6))) ^ (x3)) ^ (x7)) and (x3)) or (x2)) ^ ((((x5) or (x5)) and (not (x0))) and (x9)))) ^ (not ((x9) ^ ((x10) and (x6)))))) or (((x5) or (x5)) and (not (x0)))) or (x4)) ^ ((x2) ^ (x5))) and (x8)) ^ (((x0) or (x0)) or (x5))))
    
def pf_v11_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return not (not ((((((not ((not (((((((((((not (((x6) or (x5)) or (x7))) ^ (x7)) or (x8)) and (x3)) and (x7)) and ((x1) or (x6))) ^ (x3)) ^ (x7)) and (x3)) or (x2)) ^ ((((x5) or (x5)) and (not (x0))) and (x9)))) ^ (not ((x9) ^ ((x10) and (x6)))))) or (((x5) or (x5)) and (not (x0)))) or (x4)) ^ ((x2) ^ (x5))) and (x8)) ^ (((x0) or (x0)) or (x5))))
    

@classical_function
def cf_v11_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return (((not (not (((((((((((not ((not (((((not ((x10) and (x7))) ^ ((x3) and (x0))) or (x0)) or (not (x9))) ^ ((x2) or (x2)))) and ((x6) or (x7)))) or (x7)) or ((x3) and (x0))) ^ ((x10) or (x1))) ^ (x5)) or (not (x7))) and (not (x9))) ^ (not (x8))) ^ (((x2) or (x2)) and (x8))) or ((not (x8)) ^ (x10))) and (((x2) or (x2)) ^ (x6))))) and (x8)) and (x4)) or ((x3) ^ (not (x9)))
    
def pf_v11_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (((not (not (((((((((((not ((not (((((not ((x10) and (x7))) ^ ((x3) and (x0))) or (x0)) or (not (x9))) ^ ((x2) or (x2)))) and ((x6) or (x7)))) or (x7)) or ((x3) and (x0))) ^ ((x10) or (x1))) ^ (x5)) or (not (x7))) and (not (x9))) ^ (not (x8))) ^ (((x2) or (x2)) and (x8))) or ((not (x8)) ^ (x10))) and (((x2) or (x2)) ^ (x6))))) and (x8)) and (x4)) or ((x3) ^ (not (x9)))
    

@classical_function
def cf_v11_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1) -> Int1:
    return ((((((((((((not (((((((((not ((((x7) and (x6)) or ((x2) and (x3))) or (x1))) ^ (x5)) and (x4)) or ((x5) or (x6))) and ((x5) or (x6))) and ((x2) and (x3))) ^ (x2)) or ((x2) and (x3))) ^ (x9))) and ((x8) ^ (x0))) or (x10)) ^ (not (x8))) and ((x5) or (x6))) and ((x8) ^ (x0))) ^ (((x2) and (x3)) ^ (x9))) ^ ((x4) and (x10))) or ((((x8) or (x2)) and (x8)) ^ ((x5) or (x6)))) ^ (((x8) or (x2)) and (x8))) ^ (((x8) ^ (x0)) ^ (x9))) ^ (((x4) or (x1)) ^ (x6))) ^ (((x8) ^ (x0)) ^ (x9))
    
def pf_v11_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return ((((((((((((not (((((((((not ((((x7) and (x6)) or ((x2) and (x3))) or (x1))) ^ (x5)) and (x4)) or ((x5) or (x6))) and ((x5) or (x6))) and ((x2) and (x3))) ^ (x2)) or ((x2) and (x3))) ^ (x9))) and ((x8) ^ (x0))) or (x10)) ^ (not (x8))) and ((x5) or (x6))) and ((x8) ^ (x0))) ^ (((x2) and (x3)) ^ (x9))) ^ ((x4) and (x10))) or ((((x8) or (x2)) and (x8)) ^ ((x5) or (x6)))) ^ (((x8) or (x2)) and (x8))) ^ (((x8) ^ (x0)) ^ (x9))) ^ (((x4) or (x1)) ^ (x6))) ^ (((x8) ^ (x0)) ^ (x9))
    

@classical_function
def cf_v12_c15(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((((((((((((((not (x10)) ^ (x7)) or (x0)) or ((x5) or (x1))) and (x9)) or (x9)) ^ (x8)) ^ (((x5) or (x1)) and (not (x3)))) and (x2)) or (x4)) and (x6)) ^ (x3)) or (x8)) ^ (x9)) or (x11)
    
def pf_v12_c15(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((((((((((((((not (x10)) ^ (x7)) or (x0)) or ((x5) or (x1))) and (x9)) or (x9)) ^ (x8)) ^ (((x5) or (x1)) and (not (x3)))) and (x2)) or (x4)) and (x6)) ^ (x3)) or (x8)) ^ (x9)) or (x11)
    

@classical_function
def cf_v12_c16(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((((((((((not ((((((x4) or (x4)) and (x1)) and (x4)) or (x6)) and (x1))) ^ ((x8) and ((x0) or (x11)))) ^ (x6)) ^ (x3)) and (not (x2))) ^ (x7)) ^ ((x4) and (x9))) and (x8)) or ((x10) or ((x8) and ((x0) or (x11))))) or ((((((x10) or ((x8) and ((x0) or (x11)))) and (x8)) ^ (x1)) and (x3)) and (x5))) and (x3)
    
def pf_v12_c16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((((((((((not ((((((x4) or (x4)) and (x1)) and (x4)) or (x6)) and (x1))) ^ ((x8) and ((x0) or (x11)))) ^ (x6)) ^ (x3)) and (not (x2))) ^ (x7)) ^ ((x4) and (x9))) and (x8)) or ((x10) or ((x8) and ((x0) or (x11))))) or ((((((x10) or ((x8) and ((x0) or (x11)))) and (x8)) ^ (x1)) and (x3)) and (x5))) and (x3)
    

@classical_function
def cf_v12_c17(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((not (((((((((((((not ((x7) ^ (x5))) and (x3)) or (x1)) and (x1)) or (x11)) and (x5)) and (x0)) or ((x2) and (x3))) or (((x5) ^ (x10)) and (x4))) and (x7)) and (x2)) ^ ((x6) or (x5))) or ((((x5) ^ (x10)) or (not (x10))) and (x9)))) ^ (x7)) or (x8)
    
def pf_v12_c17(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((not (((((((((((((not ((x7) ^ (x5))) and (x3)) or (x1)) and (x1)) or (x11)) and (x5)) and (x0)) or ((x2) and (x3))) or (((x5) ^ (x10)) and (x4))) and (x7)) and (x2)) ^ ((x6) or (x5))) or ((((x5) ^ (x10)) or (not (x10))) and (x9)))) ^ (x7)) or (x8)
    

@classical_function
def cf_v12_c18(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return (((((((not ((not ((((not (((((x6) ^ (x4)) or (x11)) or (x3)) or (x7))) and (x1)) or ((x0) or ((x11) or (x11)))) ^ ((x5) or (x7)))) or ((x10) ^ (x6)))) ^ (x7)) and ((x4) or (x6))) or ((x0) or ((x11) or (x11)))) or (((x9) and (x2)) ^ ((x6) or (x11)))) or (x8)) or (((x9) and (x2)) ^ ((x6) or (x11)))) ^ (x10)
    
def pf_v12_c18(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (((((((not ((not ((((not (((((x6) ^ (x4)) or (x11)) or (x3)) or (x7))) and (x1)) or ((x0) or ((x11) or (x11)))) ^ ((x5) or (x7)))) or ((x10) ^ (x6)))) ^ (x7)) and ((x4) or (x6))) or ((x0) or ((x11) or (x11)))) or (((x9) and (x2)) ^ ((x6) or (x11)))) or (x8)) or (((x9) and (x2)) ^ ((x6) or (x11)))) ^ (x10)
    

@classical_function
def cf_v12_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((not (((((((((((((((((x1) or (x11)) ^ (x1)) or (x0)) or (x6)) and (x8)) ^ ((x5) or (x4))) ^ (x1)) ^ ((x1) or (x2))) and (x7)) or (x1)) and (x7)) ^ (x10)) or ((x1) and ((x4) ^ (x6)))) or ((x6) or (((x10) ^ ((x5) or (x4))) or (x5)))) or ((x1) and ((x4) ^ (x6)))) and (x3))) and (x6)) ^ (x9)
    
def pf_v12_c19(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((not (((((((((((((((((x1) or (x11)) ^ (x1)) or (x0)) or (x6)) and (x8)) ^ ((x5) or (x4))) ^ (x1)) ^ ((x1) or (x2))) and (x7)) or (x1)) and (x7)) ^ (x10)) or ((x1) and ((x4) ^ (x6)))) or ((x6) or (((x10) ^ ((x5) or (x4))) or (x5)))) or ((x1) and ((x4) ^ (x6)))) and (x3))) and (x6)) ^ (x9)
    

@classical_function
def cf_v12_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((not (((not ((((((((not (((((((x0) ^ (x9)) ^ (x7)) ^ (x4)) and (x3)) or (x0)) and (x0))) and (x6)) ^ (x2)) or ((x11) ^ (x3))) or (x7)) or (x0)) and (x5)) ^ ((x2) ^ (x8)))) or ((x0) ^ (not (x8)))) ^ (x11))) ^ (((x7) and (x1)) or (x0))) or ((x8) ^ (x10))
    
def pf_v12_c20(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((not (((not ((((((((not (((((((x0) ^ (x9)) ^ (x7)) ^ (x4)) and (x3)) or (x0)) and (x0))) and (x6)) ^ (x2)) or ((x11) ^ (x3))) or (x7)) or (x0)) and (x5)) ^ ((x2) ^ (x8)))) or ((x0) ^ (not (x8)))) ^ (x11))) ^ (((x7) and (x1)) or (x0))) or ((x8) ^ (x10))
    

@classical_function
def cf_v12_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return (not (not (((((not ((((((((not (((((not (x11)) or (x4)) or (x9)) and ((x0) or (x3))) and (x2))) or (x5)) ^ ((x7) and (x11))) or (x1)) and (x5)) or (x4)) and ((x0) or (x3))) and ((x7) and (x10)))) or ((x0) and (x8))) ^ ((x0) and (x8))) ^ ((x2) or (x6))) ^ (not ((x2) or (x6)))))) ^ (x0)
    
def pf_v12_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (not (not (((((not ((((((((not (((((not (x11)) or (x4)) or (x9)) and ((x0) or (x3))) and (x2))) or (x5)) ^ ((x7) and (x11))) or (x1)) and (x5)) or (x4)) and ((x0) or (x3))) and ((x7) and (x10)))) or ((x0) and (x8))) ^ ((x0) and (x8))) ^ ((x2) or (x6))) ^ (not ((x2) or (x6)))))) ^ (x0)
    

@classical_function
def cf_v12_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((((not (not (not (((not ((((not (((((((not ((x1) and (x10))) or (x5)) and (x3)) and ((x4) or (x1))) and (x9)) ^ (x10)) ^ ((x11) and (not (x9))))) or ((x2) ^ (x2))) or (x6)) or ((x2) ^ (x2)))) or (x1)) ^ (not (x9)))))) or ((x1) ^ ((x4) or (x1)))) and (x8)) ^ ((x7) or (x0))) and ((x6) and ((x11) and (not (x9))))
    
def pf_v12_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((((not (not (not (((not ((((not (((((((not ((x1) and (x10))) or (x5)) and (x3)) and ((x4) or (x1))) and (x9)) ^ (x10)) ^ ((x11) and (not (x9))))) or ((x2) ^ (x2))) or (x6)) or ((x2) ^ (x2)))) or (x1)) ^ (not (x9)))))) or ((x1) ^ ((x4) or (x1)))) and (x8)) ^ ((x7) or (x0))) and ((x6) and ((x11) and (not (x9))))
    

@classical_function
def cf_v12_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return (not (not (not ((((not ((((not ((((((((((((x3) ^ (x5)) and (x9)) ^ (x0)) and (x6)) ^ (x7)) and (x6)) ^ (x3)) or (x8)) or (x3)) or (((x2) ^ (x9)) and (x10))) and (x9))) or (((x2) ^ (x9)) ^ (x4))) or (x7)) or (x5))) ^ ((((x2) ^ (x11)) or (x4)) and (x4))) or (x2)) or ((x2) ^ (x11)))))) ^ (x11)
    
def pf_v12_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (not (not (not ((((not ((((not ((((((((((((x3) ^ (x5)) and (x9)) ^ (x0)) and (x6)) ^ (x7)) and (x6)) ^ (x3)) or (x8)) or (x3)) or (((x2) ^ (x9)) and (x10))) and (x9))) or (((x2) ^ (x9)) ^ (x4))) or (x7)) or (x5))) ^ ((((x2) ^ (x11)) or (x4)) and (x4))) or (x2)) or ((x2) ^ (x11)))))) ^ (x11)
    

@classical_function
def cf_v12_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return not (not ((((((((((((((((not (not ((((((x10) or (x8)) or (x5)) and (x6)) or (x0)) ^ (x4)))) or (not (x9))) ^ ((x7) ^ (x3))) or (x4)) or (x9)) and (not ((x2) and ((x5) or (x3))))) and (x0)) and (x7)) or (x0)) or ((x5) or (x7))) ^ (((x5) or (x3)) or (x10))) or ((x5) or (x3))) ^ ((x11) and (x5))) ^ (not (x9))) or ((x5) or (x7))) and ((x9) ^ (x9))))
    
def pf_v12_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return not (not ((((((((((((((((not (not ((((((x10) or (x8)) or (x5)) and (x6)) or (x0)) ^ (x4)))) or (not (x9))) ^ ((x7) ^ (x3))) or (x4)) or (x9)) and (not ((x2) and ((x5) or (x3))))) and (x0)) and (x7)) or (x0)) or ((x5) or (x7))) ^ (((x5) or (x3)) or (x10))) or ((x5) or (x3))) ^ ((x11) and (x5))) ^ (not (x9))) or ((x5) or (x7))) and ((x9) ^ (x9))))
    

@classical_function
def cf_v12_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((((not (((((((not ((((not ((not (((((((not (x10)) and (x6)) and (x9)) ^ (x8)) and (x1)) and (x0)) and ((x0) and (x2)))) and (not (x1)))) ^ ((x6) and (x11))) ^ (x4)) or (x1))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (x7)) or (x3)) and ((not (x1)) ^ ((x8) and (x11)))) ^ ((not (x6)) ^ (x1)))) or ((x0) and (x2))) or ((x6) and (x11))) ^ ((x6) and (x11))) and ((x5) and (x6))
    
def pf_v12_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((((not (((((((not ((((not ((not (((((((not (x10)) and (x6)) and (x9)) ^ (x8)) and (x1)) and (x0)) and ((x0) and (x2)))) and (not (x1)))) ^ ((x6) and (x11))) ^ (x4)) or (x1))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (x7)) or (x3)) and ((not (x1)) ^ ((x8) and (x11)))) ^ ((not (x6)) ^ (x1)))) or ((x0) and (x2))) or ((x6) and (x11))) ^ ((x6) and (x11))) and ((x5) and (x6))
    

@classical_function
def cf_v12_c26(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1) -> Int1:
    return ((((((((not (not ((not ((not ((((((not ((not ((((not (x10)) or (x5)) and (x5)) or ((x0) and (x4)))) ^ (x1))) ^ (x1)) and ((x5) or (x3))) or ((x3) or (x11))) and (x11)) ^ ((x3) or (x11)))) ^ (x8))) and ((x4) ^ (x11))))) and (x2)) or ((x7) ^ (x8))) or (x3)) or (x6)) ^ (((x5) ^ (not (x0))) or (((x5) ^ (not (x0))) or ((x5) or (x3))))) and (x8)) ^ (x10)) or (x9)
    
def pf_v12_c26(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((((((((not (not ((not ((not ((((((not ((not ((((not (x10)) or (x5)) and (x5)) or ((x0) and (x4)))) ^ (x1))) ^ (x1)) and ((x5) or (x3))) or ((x3) or (x11))) and (x11)) ^ ((x3) or (x11)))) ^ (x8))) and ((x4) ^ (x11))))) and (x2)) or ((x7) ^ (x8))) or (x3)) or (x6)) ^ (((x5) ^ (not (x0))) or (((x5) ^ (not (x0))) or ((x5) or (x3))))) and (x8)) ^ (x10)) or (x9)
    

@classical_function
def cf_v13_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return (((((((((not (((((not (((((((x4) ^ (x10)) and ((x4) ^ (x2))) or (x6)) ^ ((x4) ^ (x2))) ^ (x0)) ^ (x8))) ^ (x11)) and (x6)) and ((x5) ^ ((x10) or (x5)))) and (x11))) ^ (x7)) or (not (x2))) and (x0)) or (((x1) ^ (x11)) and ((x5) ^ ((x10) or (x5))))) or (x9)) ^ (x4)) ^ ((x5) and (x3))) or (x12)) ^ (x9)
    
def pf_v13_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return (((((((((not (((((not (((((((x4) ^ (x10)) and ((x4) ^ (x2))) or (x6)) ^ ((x4) ^ (x2))) ^ (x0)) ^ (x8))) ^ (x11)) and (x6)) and ((x5) ^ ((x10) or (x5)))) and (x11))) ^ (x7)) or (not (x2))) and (x0)) or (((x1) ^ (x11)) and ((x5) ^ ((x10) or (x5))))) or (x9)) ^ (x4)) ^ ((x5) and (x3))) or (x12)) ^ (x9)
    

@classical_function
def cf_v13_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return (((((not (((not (((not (((((not ((((((x4) and (x5)) ^ (x7)) ^ (x10)) and (x3)) or (x6))) ^ (x2)) and (x4)) or ((x4) ^ (x0))) ^ (x11))) or (x8)) ^ (x12))) ^ (x7)) or (((x10) ^ (x2)) ^ ((x3) or (x7))))) and ((x10) ^ (x2))) and ((x4) or (x11))) and ((x1) and (x6))) and (x9)) and (x7)
    
def pf_v13_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return (((((not (((not (((not (((((not ((((((x4) and (x5)) ^ (x7)) ^ (x10)) and (x3)) or (x6))) ^ (x2)) and (x4)) or ((x4) ^ (x0))) ^ (x11))) or (x8)) ^ (x12))) ^ (x7)) or (((x10) ^ (x2)) ^ ((x3) or (x7))))) and ((x10) ^ (x2))) and ((x4) or (x11))) and ((x1) and (x6))) and (x9)) and (x7)
    

@classical_function
def cf_v13_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return ((((not ((not ((((((not (not (not (not (not ((not (((((x0) and (x4)) or (x5)) and ((x10) or (x11))) and (x6))) or (x11))))))) and (x10)) or ((x6) and ((x6) and (x9)))) or (x12)) or (not ((x8) and (x11)))) and (not ((x10) or (x11))))) or ((x6) and ((x6) and (x9))))) ^ ((not (x7)) or (x12))) ^ ((x3) or (x12))) and (not (x8))) ^ ((x2) or (x8))
    
def pf_v13_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return ((((not ((not ((((((not (not (not (not (not ((not (((((x0) and (x4)) or (x5)) and ((x10) or (x11))) and (x6))) or (x11))))))) and (x10)) or ((x6) and ((x6) and (x9)))) or (x12)) or (not ((x8) and (x11)))) and (not ((x10) or (x11))))) or ((x6) and ((x6) and (x9))))) ^ ((not (x7)) or (x12))) ^ ((x3) or (x12))) and (not (x8))) ^ ((x2) or (x8))
    

@classical_function
def cf_v13_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return not (((((not ((((((((not ((((not (not (not (((not ((x8) and (x7))) ^ ((x9) and (x3))) and (x4))))) or (x10)) ^ (x7)) and (x2))) ^ (not (x5))) ^ (not (x8))) and (x0)) ^ (x12)) or (not (x11))) ^ (not (x6))) or (((x10) or (x8)) and (x7)))) ^ (((x10) or (x8)) and (x7))) ^ (x2)) ^ (x5)) or (x3))
    
def pf_v13_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return not (((((not ((((((((not ((((not (not (not (((not ((x8) and (x7))) ^ ((x9) and (x3))) and (x4))))) or (x10)) ^ (x7)) and (x2))) ^ (not (x5))) ^ (not (x8))) and (x0)) ^ (x12)) or (not (x11))) ^ (not (x6))) or (((x10) or (x8)) and (x7)))) ^ (((x10) or (x8)) and (x7))) ^ (x2)) ^ (x5)) or (x3))
    

@classical_function
def cf_v13_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return ((not (((((((((not (not ((((((not ((((((not (x5)) or (x3)) and (x6)) or (x4)) ^ (x12)) and (x0))) or (x2)) and (x7)) or (((x6) and ((x4) or (x11))) or (x4))) and ((x6) or (x1))) ^ (x5)))) ^ (((x6) and ((x4) or (x11))) and ((x6) and ((x4) or (x11))))) and ((x6) or (x1))) and (((x4) or (x11)) ^ (x8))) or ((x0) and ((x11) ^ (x12)))) and (x3)) or (not (x9))) and ((x0) and ((x11) ^ (x12)))) ^ (x5))) and ((x5) and (not (x12)))) and ((x9) and ((((x4) or (x11)) ^ (x8)) or (((x4) or (x11)) and (x10))))
    
def pf_v13_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return ((not (((((((((not (not ((((((not ((((((not (x5)) or (x3)) and (x6)) or (x4)) ^ (x12)) and (x0))) or (x2)) and (x7)) or (((x6) and ((x4) or (x11))) or (x4))) and ((x6) or (x1))) ^ (x5)))) ^ (((x6) and ((x4) or (x11))) and ((x6) and ((x4) or (x11))))) and ((x6) or (x1))) and (((x4) or (x11)) ^ (x8))) or ((x0) and ((x11) ^ (x12)))) and (x3)) or (not (x9))) and ((x0) and ((x11) ^ (x12)))) ^ (x5))) and ((x5) and (not (x12)))) and ((x9) and ((((x4) or (x11)) ^ (x8)) or (((x4) or (x11)) and (x10))))
    

@classical_function
def cf_v13_c26(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return not (((not (((((((((((not ((not (((not (((((((x8) ^ (x7)) and (x1)) ^ (x11)) or (x6)) ^ (x3)) or (x6))) ^ ((x6) ^ (x1))) and ((x11) or (x11)))) and ((x6) ^ (x1)))) ^ (x7)) and ((x11) or (x11))) ^ (x8)) ^ ((x2) and (x10))) or (x3)) ^ (x12)) or (((x6) ^ (x1)) ^ (x9))) and (x12)) or (((x4) and (x9)) ^ (x11))) or (not (x11)))) ^ (((x6) or ((x5) ^ (x5))) and (not (x6)))) ^ (((x6) ^ (x1)) and (x0)))
    
def pf_v13_c26(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return not (((not (((((((((((not ((not (((not (((((((x8) ^ (x7)) and (x1)) ^ (x11)) or (x6)) ^ (x3)) or (x6))) ^ ((x6) ^ (x1))) and ((x11) or (x11)))) and ((x6) ^ (x1)))) ^ (x7)) and ((x11) or (x11))) ^ (x8)) ^ ((x2) and (x10))) or (x3)) ^ (x12)) or (((x6) ^ (x1)) ^ (x9))) and (x12)) or (((x4) and (x9)) ^ (x11))) or (not (x11)))) ^ (((x6) or ((x5) ^ (x5))) and (not (x6)))) ^ (((x6) ^ (x1)) and (x0)))
    

@classical_function
def cf_v13_c27(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1) -> Int1:
    return not (((((((((not ((((((not (((((not ((((not (not ((x0) ^ (x4)))) ^ (x3)) or (not (x8))) or (x11))) ^ ((x9) or (x7))) ^ (x3)) ^ ((x9) or (x7))) ^ (x12))) or (not (x2))) and (x5)) and (x6)) and ((x1) ^ (x9))) or (x8))) ^ (x2)) and ((x3) or ((x10) ^ (x4)))) and (x5)) or (x10)) or (x10)) or ((x0) or (not (not (x12))))) or (((x3) or ((x10) ^ (x4))) and (x0))) and ((x0) or (not (not (x12)))))
    
def pf_v13_c27(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return not (((((((((not ((((((not (((((not ((((not (not ((x0) ^ (x4)))) ^ (x3)) or (not (x8))) or (x11))) ^ ((x9) or (x7))) ^ (x3)) ^ ((x9) or (x7))) ^ (x12))) or (not (x2))) and (x5)) and (x6)) and ((x1) ^ (x9))) or (x8))) ^ (x2)) and ((x3) or ((x10) ^ (x4)))) and (x5)) or (x10)) or (x10)) or ((x0) or (not (not (x12))))) or (((x3) or ((x10) ^ (x4))) and (x0))) and ((x0) or (not (not (x12)))))
    

@classical_function
def cf_v14_c20(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return (((not (((((((((((((((((x7) and (x11)) ^ (x1)) ^ (x6)) ^ (x9)) and (x11)) ^ (x13)) or (x11)) and (x12)) ^ (x10)) ^ (x1)) or ((x2) or (x1))) and (x3)) or (((x9) and (x5)) ^ ((x4) ^ (x0)))) ^ ((x7) ^ (x8))) or ((x4) ^ (x6))) or (x8))) or (x4)) ^ (((x9) and (x9)) or ((x7) ^ (x8)))) ^ ((x7) and (x4))
    
def pf_v14_c20(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return (((not (((((((((((((((((x7) and (x11)) ^ (x1)) ^ (x6)) ^ (x9)) and (x11)) ^ (x13)) or (x11)) and (x12)) ^ (x10)) ^ (x1)) or ((x2) or (x1))) and (x3)) or (((x9) and (x5)) ^ ((x4) ^ (x0)))) ^ ((x7) ^ (x8))) or ((x4) ^ (x6))) or (x8))) or (x4)) ^ (((x9) and (x9)) or ((x7) ^ (x8)))) ^ ((x7) and (x4))
    

@classical_function
def cf_v14_c21(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return ((((not (((((((((((((((not ((x3) and (x5))) ^ (x0)) and (x8)) or (x8)) ^ ((x10) and (x12))) or (x13)) ^ (x6)) or (x9)) or ((x11) or ((x6) ^ (x2)))) and (x4)) ^ (x0)) ^ ((x1) and (x5))) and (x10)) or ((x7) or (x10))) or ((x11) ^ ((x5) and (x5))))) or ((x1) or ((x2) ^ (x5)))) ^ (x10)) ^ (x8)) and ((x1) ^ ((x2) or ((x2) ^ (x5))))
    
def pf_v14_c21(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return ((((not (((((((((((((((not ((x3) and (x5))) ^ (x0)) and (x8)) or (x8)) ^ ((x10) and (x12))) or (x13)) ^ (x6)) or (x9)) or ((x11) or ((x6) ^ (x2)))) and (x4)) ^ (x0)) ^ ((x1) and (x5))) and (x10)) or ((x7) or (x10))) or ((x11) ^ ((x5) and (x5))))) or ((x1) or ((x2) ^ (x5)))) ^ (x10)) ^ (x8)) and ((x1) ^ ((x2) or ((x2) ^ (x5))))
    

@classical_function
def cf_v14_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return (((((((((((((not (((((not ((((x10) ^ (x12)) ^ (x8)) and (x0))) or (x4)) ^ (x8)) ^ (x5)) or (x1))) ^ (x6)) and (x13)) and ((x3) and (x5))) ^ (x1)) and (x9)) and (x5)) ^ (((x3) and (x5)) ^ ((x13) ^ (x2)))) or ((x10) ^ (x3))) or (((x12) ^ (x11)) and (x3))) and ((x0) or ((x13) ^ (x2)))) ^ ((x13) ^ (x2))) ^ (((x10) ^ (x7)) and ((((x3) and (x5)) ^ ((x13) ^ (x2))) or (x5)))) and ((x6) ^ (not (x6)))
    
def pf_v14_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return (((((((((((((not (((((not ((((x10) ^ (x12)) ^ (x8)) and (x0))) or (x4)) ^ (x8)) ^ (x5)) or (x1))) ^ (x6)) and (x13)) and ((x3) and (x5))) ^ (x1)) and (x9)) and (x5)) ^ (((x3) and (x5)) ^ ((x13) ^ (x2)))) or ((x10) ^ (x3))) or (((x12) ^ (x11)) and (x3))) and ((x0) or ((x13) ^ (x2)))) ^ ((x13) ^ (x2))) ^ (((x10) ^ (x7)) and ((((x3) and (x5)) ^ ((x13) ^ (x2))) or (x5)))) and ((x6) ^ (not (x6)))
    

@classical_function
def cf_v14_c23(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return (not ((((not (((((((((not (not ((not (((((not (x8)) or (x11)) ^ (x2)) or (x1)) ^ (x12))) ^ (x2)))) ^ (x12)) and (x1)) or (((x1) ^ (x6)) and (x7))) ^ (x0)) and (x5)) and ((x1) ^ (x6))) ^ (x10)) or ((x11) ^ (x13)))) and (x11)) ^ (x3)) ^ (((x10) or (x11)) ^ ((not (x9)) ^ (x4))))) ^ (x11)
    
def pf_v14_c23(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return (not ((((not (((((((((not (not ((not (((((not (x8)) or (x11)) ^ (x2)) or (x1)) ^ (x12))) ^ (x2)))) ^ (x12)) and (x1)) or (((x1) ^ (x6)) and (x7))) ^ (x0)) and (x5)) and ((x1) ^ (x6))) ^ (x10)) or ((x11) ^ (x13)))) and (x11)) ^ (x3)) ^ (((x10) or (x11)) ^ ((not (x9)) ^ (x4))))) ^ (x11)
    

@classical_function
def cf_v14_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return ((((not (not (not ((((not ((((not (not (((((not ((((x6) and (x13)) or (x10)) and (x11))) or (((x10) and (x7)) or (x10))) or (x5)) ^ ((x9) or (x8))) ^ ((x10) and (x7))))) and (x11)) or (x0)) or (x13))) and (x11)) ^ ((x2) or (x4))) and ((x12) ^ (x11)))))) ^ ((x12) ^ (x11))) or (not (((x2) or (x4)) ^ (x4)))) ^ (x3)) ^ (x10)
    
def pf_v14_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return ((((not (not (not ((((not ((((not (not (((((not ((((x6) and (x13)) or (x10)) and (x11))) or (((x10) and (x7)) or (x10))) or (x5)) ^ ((x9) or (x8))) ^ ((x10) and (x7))))) and (x11)) or (x0)) or (x13))) and (x11)) ^ ((x2) or (x4))) and ((x12) ^ (x11)))))) ^ ((x12) ^ (x11))) or (not (((x2) or (x4)) ^ (x4)))) ^ (x3)) ^ (x10)
    

@classical_function
def cf_v14_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return ((((((((((((((((not ((((((not ((not (x13)) and (x3))) ^ (x12)) or (x0)) or (x8)) and ((x13) or (x12))) ^ (x4))) and (x2)) and (x2)) and (x12)) ^ ((x5) and ((x3) and (x7)))) ^ (x11)) or (not (x6))) and (x9)) ^ ((x5) and ((x3) and (x7)))) or (x4)) ^ (x2)) ^ (x0)) or ((x13) ^ ((x5) ^ ((x2) ^ (x4))))) ^ (x10)) and (not (x12))) or (not (x6))) ^ (((x5) ^ ((x2) ^ (x4))) ^ (x12))
    
def pf_v14_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return ((((((((((((((((not ((((((not ((not (x13)) and (x3))) ^ (x12)) or (x0)) or (x8)) and ((x13) or (x12))) ^ (x4))) and (x2)) and (x2)) and (x12)) ^ ((x5) and ((x3) and (x7)))) ^ (x11)) or (not (x6))) and (x9)) ^ ((x5) and ((x3) and (x7)))) or (x4)) ^ (x2)) ^ (x0)) or ((x13) ^ ((x5) ^ ((x2) ^ (x4))))) ^ (x10)) and (not (x12))) or (not (x6))) ^ (((x5) ^ ((x2) ^ (x4))) ^ (x12))
    

@classical_function
def cf_v14_c26(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return not ((((((((((((((not (((((not ((((((not (x4)) or (x2)) or (x3)) and ((x7) or (x8))) ^ (x1)) and (x0))) ^ (x6)) ^ (x4)) ^ (x12)) ^ (x0))) or (x10)) or (x8)) ^ (x2)) ^ (((x7) or (x8)) ^ ((x3) or (x5)))) ^ ((x13) or (not (((not (x8)) and ((x3) or (x5))) ^ ((not (x8)) and ((x3) or (x5))))))) ^ (x3)) and (not (x8))) or ((x7) or (x8))) and ((x2) ^ (x13))) ^ ((x0) ^ (x11))) ^ ((x9) or ((x0) and (x13)))) ^ (x3)) ^ (x13))
    
def pf_v14_c26(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return not ((((((((((((((not (((((not ((((((not (x4)) or (x2)) or (x3)) and ((x7) or (x8))) ^ (x1)) and (x0))) ^ (x6)) ^ (x4)) ^ (x12)) ^ (x0))) or (x10)) or (x8)) ^ (x2)) ^ (((x7) or (x8)) ^ ((x3) or (x5)))) ^ ((x13) or (not (((not (x8)) and ((x3) or (x5))) ^ ((not (x8)) and ((x3) or (x5))))))) ^ (x3)) and (not (x8))) or ((x7) or (x8))) and ((x2) ^ (x13))) ^ ((x0) ^ (x11))) ^ ((x9) or ((x0) and (x13)))) ^ (x3)) ^ (x13))
    

@classical_function
def cf_v14_c27(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return ((not (not (not ((not ((((((not (((not (((((((not (((((x13) ^ (x4)) ^ (x11)) ^ (x7)) or (x8))) ^ (x8)) and (x0)) or ((x9) ^ (x2))) ^ (x0)) ^ (x12)) and ((x9) or (x10)))) and (x3)) ^ (x6))) ^ (x5)) and (not (((x9) ^ (x2)) and (x8)))) ^ (not (((x9) ^ (x2)) and (x8)))) or ((x4) and (x4))) or ((x7) ^ ((x7) ^ (x12))))) ^ (x9))))) and (x5)) and ((x3) or ((x2) ^ (x1)))
    
def pf_v14_c27(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return ((not (not (not ((not ((((((not (((not (((((((not (((((x13) ^ (x4)) ^ (x11)) ^ (x7)) or (x8))) ^ (x8)) and (x0)) or ((x9) ^ (x2))) ^ (x0)) ^ (x12)) and ((x9) or (x10)))) and (x3)) ^ (x6))) ^ (x5)) and (not (((x9) ^ (x2)) and (x8)))) ^ (not (((x9) ^ (x2)) and (x8)))) or ((x4) and (x4))) or ((x7) ^ ((x7) ^ (x12))))) ^ (x9))))) and (x5)) and ((x3) or ((x2) ^ (x1)))
    

@classical_function
def cf_v14_c28(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1) -> Int1:
    return (not (((((((((((((((((not (((not (not ((((((x10) or (x1)) or (x0)) or (x3)) or (x12)) or (x12)))) ^ ((x5) or ((x11) or (x2)))) ^ (x10))) or (((x6) or (x3)) or (x11))) and (not (x1))) or ((not ((x7) ^ (x7))) and (not (x1)))) ^ (x9)) ^ (x4)) or (not (x13))) or (x0)) and (x11)) or ((not (x13)) or (x7))) or ((x6) or (x3))) or ((x8) or (x1))) ^ (x7)) and (x2)) and (not (x1))) ^ ((x4) or (x4))) or (x0))) and ((x6) or (x3))
    
def pf_v14_c28(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    return (not (((((((((((((((((not (((not (not ((((((x10) or (x1)) or (x0)) or (x3)) or (x12)) or (x12)))) ^ ((x5) or ((x11) or (x2)))) ^ (x10))) or (((x6) or (x3)) or (x11))) and (not (x1))) or ((not ((x7) ^ (x7))) and (not (x1)))) ^ (x9)) ^ (x4)) or (not (x13))) or (x0)) and (x11)) or ((not (x13)) or (x7))) or ((x6) or (x3))) or ((x8) or (x1))) ^ (x7)) and (x2)) and (not (x1))) ^ ((x4) or (x4))) or (x0))) and ((x6) or (x3))
    

@classical_function
def cf_v15_c19(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return ((((not (((((((not ((((((((x5) and (x10)) or (x8)) and (x9)) ^ ((x3) or (x2))) ^ (x13)) or ((x11) and (x12))) ^ (x14))) and (x10)) or ((x13) or (x1))) ^ (x7)) ^ ((x13) ^ (x0))) ^ ((x13) ^ (x0))) and ((x13) ^ (x0)))) or (x4)) or ((x4) ^ (x7))) and ((x6) or (x12))) or (x11)
    
def pf_v15_c19(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return ((((not (((((((not ((((((((x5) and (x10)) or (x8)) and (x9)) ^ ((x3) or (x2))) ^ (x13)) or ((x11) and (x12))) ^ (x14))) and (x10)) or ((x13) or (x1))) ^ (x7)) ^ ((x13) ^ (x0))) ^ ((x13) ^ (x0))) and ((x13) ^ (x0)))) or (x4)) or ((x4) ^ (x7))) and ((x6) or (x12))) or (x11)
    

@classical_function
def cf_v15_c22(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return (not (((((((((not (not (not (((((((not (((x10) ^ (x10)) and ((x12) ^ (x9)))) or (x13)) and (x7)) ^ (not (x3))) and (x6)) and (x8)) and (x6))))) and ((not (x3)) and (((x2) and (x11)) and (x5)))) ^ ((x2) and (x11))) or ((x2) and (x11))) and (x14)) or (not (x0))) ^ (not (x11))) ^ (x4)) and (x9))) and (not (x8))
    
def pf_v15_c22(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return (not (((((((((not (not (not (((((((not (((x10) ^ (x10)) and ((x12) ^ (x9)))) or (x13)) and (x7)) ^ (not (x3))) and (x6)) and (x8)) and (x6))))) and ((not (x3)) and (((x2) and (x11)) and (x5)))) ^ ((x2) and (x11))) or ((x2) and (x11))) and (x14)) or (not (x0))) ^ (not (x11))) ^ (x4)) and (x9))) and (not (x8))
    

@classical_function
def cf_v15_c26(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return (((((((not ((not (not (not ((((((((((not ((not (((x7) or (x6)) and (x3))) or (x5))) and (x14)) or (x5)) and (x11)) or (x13)) or (x10)) or ((x9) or (x5))) ^ (x8)) ^ ((x2) or ((x11) or (x1)))) and (x0))))) and (x1))) or (x10)) ^ (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) ^ (x0)) and (x3)) or (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) and (((not ((x12) and (x14))) and (not ((x12) and (x14)))) or (x10))) and (((x2) ^ ((x12) and (x6))) or ((x11) or (x1)))
    
def pf_v15_c26(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return (((((((not ((not (not (not ((((((((((not ((not (((x7) or (x6)) and (x3))) or (x5))) and (x14)) or (x5)) and (x11)) or (x13)) or (x10)) or ((x9) or (x5))) ^ (x8)) ^ ((x2) or ((x11) or (x1)))) and (x0))))) and (x1))) or (x10)) ^ (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) ^ (x0)) and (x3)) or (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) and (((not ((x12) and (x14))) and (not ((x12) and (x14)))) or (x10))) and (((x2) ^ ((x12) and (x6))) or ((x11) or (x1)))
    

@classical_function
def cf_v15_c27(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return ((((not (((not ((not ((((not ((((not (((((not ((not (((x8) or (x10)) or (x3))) or (x5))) ^ (x10)) or (not (x6))) and ((x12) or ((x2) or (x14)))) and ((x9) and (x9)))) ^ (not (x14))) or (x2)) or (x8))) and ((x7) or (x8))) ^ (x1)) ^ ((x13) ^ (x14)))) and ((x12) or ((x2) or (x14))))) and ((x5) or (x11))) and (x4))) ^ (x0)) ^ (x0)) ^ ((x4) ^ (not (x6)))) ^ ((x3) or ((x12) or ((x2) or (x14))))
    
def pf_v15_c27(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return ((((not (((not ((not ((((not ((((not (((((not ((not (((x8) or (x10)) or (x3))) or (x5))) ^ (x10)) or (not (x6))) and ((x12) or ((x2) or (x14)))) and ((x9) and (x9)))) ^ (not (x14))) or (x2)) or (x8))) and ((x7) or (x8))) ^ (x1)) ^ ((x13) ^ (x14)))) and ((x12) or ((x2) or (x14))))) and ((x5) or (x11))) and (x4))) ^ (x0)) ^ (x0)) ^ ((x4) ^ (not (x6)))) ^ ((x3) or ((x12) or ((x2) or (x14))))
    

@classical_function
def cf_v15_c28(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return (((not ((((not (((((not ((((((((not (not (((((not ((x3) ^ (x2))) ^ (x9)) and (x4)) and (x9)) and (x0)))) ^ (x5)) and (x0)) ^ ((x3) and (x7))) and (x1)) or (x13)) or ((x5) ^ (x0))) ^ ((not (x4)) and (x8)))) or (x10)) and ((x6) and (x4))) and ((x9) ^ (x12))) and ((not (x0)) and (x1)))) and (not (x4))) ^ (x7)) and (x7))) and (not (x11))) and (x0)) ^ (x14)
    
def pf_v15_c28(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return (((not ((((not (((((not ((((((((not (not (((((not ((x3) ^ (x2))) ^ (x9)) and (x4)) and (x9)) and (x0)))) ^ (x5)) and (x0)) ^ ((x3) and (x7))) and (x1)) or (x13)) or ((x5) ^ (x0))) ^ ((not (x4)) and (x8)))) or (x10)) and ((x6) and (x4))) and ((x9) ^ (x12))) and ((not (x0)) and (x1)))) and (not (x4))) ^ (x7)) and (x7))) and (not (x11))) and (x0)) ^ (x14)
    

@classical_function
def cf_v15_c29(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1) -> Int1:
    return ((((not (not ((((((((not ((not ((((not ((((((((((x5) ^ (x14)) and (x6)) ^ (x7)) or (x9)) ^ (x8)) and (x13)) and ((x5) or (x9))) and ((x5) or (x9))) ^ (x11))) ^ ((x10) ^ (x7))) ^ (x6)) ^ ((x5) or (x9)))) ^ (not (x0)))) and ((x5) or (x9))) and ((x12) ^ ((x5) or (x9)))) ^ ((x9) ^ ((x2) ^ (x11)))) ^ (x4)) or (not (x2))) or ((x8) and ((x2) ^ (x11)))) or (((x14) ^ (x3)) ^ (not (x0)))))) or (x5)) ^ (x13)) or ((x1) ^ (not (x10)))) or ((x12) ^ ((x5) or (x9)))
    
def pf_v15_c29(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    return ((((not (not ((((((((not ((not ((((not ((((((((((x5) ^ (x14)) and (x6)) ^ (x7)) or (x9)) ^ (x8)) and (x13)) and ((x5) or (x9))) and ((x5) or (x9))) ^ (x11))) ^ ((x10) ^ (x7))) ^ (x6)) ^ ((x5) or (x9)))) ^ (not (x0)))) and ((x5) or (x9))) and ((x12) ^ ((x5) or (x9)))) ^ ((x9) ^ ((x2) ^ (x11)))) ^ (x4)) or (not (x2))) or ((x8) and ((x2) ^ (x11)))) or (((x14) ^ (x3)) ^ (not (x0)))))) or (x5)) ^ (x13)) or ((x1) ^ (not (x10)))) or ((x12) ^ ((x5) or (x9)))
    

@classical_function
def cf_v16_c24(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return ((((((((((((((((((((((((x13) ^ (x8)) and (x13)) and (x11)) and (x2)) ^ (x12)) and (x2)) ^ ((x3) and (x9))) or ((x7) and ((x10) and (x14)))) or ((x13) ^ (x9))) and (x10)) or (x6)) and (x8)) or (x6)) or (x1)) and (x14)) ^ (x12)) or (((x15) ^ ((x13) ^ (x9))) and (x5))) ^ (x7)) ^ (x3)) and (x0)) and (x7)) ^ ((((x15) ^ ((x13) ^ (x9))) and (x5)) ^ (((x15) ^ ((x13) ^ (x9))) and (x5)))) ^ (x3)) and ((x4) ^ ((x2) or (x15)))
    
def pf_v16_c24(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return ((((((((((((((((((((((((x13) ^ (x8)) and (x13)) and (x11)) and (x2)) ^ (x12)) and (x2)) ^ ((x3) and (x9))) or ((x7) and ((x10) and (x14)))) or ((x13) ^ (x9))) and (x10)) or (x6)) and (x8)) or (x6)) or (x1)) and (x14)) ^ (x12)) or (((x15) ^ ((x13) ^ (x9))) and (x5))) ^ (x7)) ^ (x3)) and (x0)) and (x7)) ^ ((((x15) ^ ((x13) ^ (x9))) and (x5)) ^ (((x15) ^ ((x13) ^ (x9))) and (x5)))) ^ (x3)) and ((x4) ^ ((x2) or (x15)))
    

@classical_function
def cf_v16_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return (((((((((((((((not ((((not ((not ((((x5) or (x14)) and ((x4) or (x4))) and (x2))) ^ (x12))) or (x14)) and ((x3) and (x4))) and (x0))) ^ (x8)) ^ (x14)) ^ (x6)) or ((x15) or ((x2) ^ (x8)))) and (x15)) ^ ((x9) ^ (x10))) ^ (x14)) and ((x4) or (x4))) and (not (x13))) or (x3)) ^ ((((x7) and ((x4) or (x4))) or ((x4) or (x4))) ^ (x4))) ^ (x15)) ^ ((x9) ^ (x10))) or (x11)) or ((x3) and (((x15) or ((x2) ^ (x8))) ^ ((x7) and ((x4) or (x4)))))
    
def pf_v16_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return (((((((((((((((not ((((not ((not ((((x5) or (x14)) and ((x4) or (x4))) and (x2))) ^ (x12))) or (x14)) and ((x3) and (x4))) and (x0))) ^ (x8)) ^ (x14)) ^ (x6)) or ((x15) or ((x2) ^ (x8)))) and (x15)) ^ ((x9) ^ (x10))) ^ (x14)) and ((x4) or (x4))) and (not (x13))) or (x3)) ^ ((((x7) and ((x4) or (x4))) or ((x4) or (x4))) ^ (x4))) ^ (x15)) ^ ((x9) ^ (x10))) or (x11)) or ((x3) and (((x15) or ((x2) ^ (x8))) ^ ((x7) and ((x4) or (x4)))))
    

@classical_function
def cf_v16_c26(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return ((not (((((((not (not ((((((((((((((not (not (x15))) and (x9)) ^ (x3)) or (x9)) and ((x4) or (x15))) or (x12)) or (x2)) and (x7)) and (x0)) or ((x0) or (not (x4)))) or ((x0) or (not (x4)))) or ((x1) ^ (x0))) or (x10)) ^ ((x4) or (x13))))) ^ (x4)) and ((not (x4)) and (x13))) and (x11)) and (((x8) or (x1)) or (x9))) and (x14)) and (not (x13)))) and ((x10) and (x6))) ^ ((x5) and ((x1) ^ (x0)))
    
def pf_v16_c26(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return ((not (((((((not (not ((((((((((((((not (not (x15))) and (x9)) ^ (x3)) or (x9)) and ((x4) or (x15))) or (x12)) or (x2)) and (x7)) and (x0)) or ((x0) or (not (x4)))) or ((x0) or (not (x4)))) or ((x1) ^ (x0))) or (x10)) ^ ((x4) or (x13))))) ^ (x4)) and ((not (x4)) and (x13))) and (x11)) and (((x8) or (x1)) or (x9))) and (x14)) and (not (x13)))) and ((x10) and (x6))) ^ ((x5) and ((x1) ^ (x0)))
    

@classical_function
def cf_v16_c27(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return ((((((((((((((not (not (((((not ((((not (((x4) or (x7)) ^ (x2))) and (x8)) and (x7)) and (x7))) or (x14)) ^ (x12)) and (x13)) and (((x12) and (x15)) ^ (x0))))) or (x12)) ^ (((x15) ^ (x10)) and (not (x4)))) and (x2)) and (x8)) or (x10)) and ((x9) and (x5))) or (x15)) and (not (x12))) or (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (((x5) and (x15)) or (x2))) and (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (x11)) or ((x12) and (x6))) ^ (x5)
    
def pf_v16_c27(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return ((((((((((((((not (not (((((not ((((not (((x4) or (x7)) ^ (x2))) and (x8)) and (x7)) and (x7))) or (x14)) ^ (x12)) and (x13)) and (((x12) and (x15)) ^ (x0))))) or (x12)) ^ (((x15) ^ (x10)) and (not (x4)))) and (x2)) and (x8)) or (x10)) and ((x9) and (x5))) or (x15)) and (not (x12))) or (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (((x5) and (x15)) or (x2))) and (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (x11)) or ((x12) and (x6))) ^ (x5)
    

@classical_function
def cf_v16_c28(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return (not (not ((not ((((((not ((((not ((not ((((((not (((((not (x9)) and (x14)) and (x8)) ^ (x2)) ^ (x5))) and (x13)) or (x1)) and (not (not (x3)))) ^ (x13)) ^ (x5))) ^ ((x15) or (x6)))) and (x11)) and (x9)) or (x10))) and ((x6) ^ (x4))) or (x11)) and (x12)) and ((x2) and (((x3) and (x0)) ^ (x11)))) ^ (((x15) and (x1)) and ((x7) ^ (x8))))) and ((x15) or (((x7) ^ (x8)) ^ (x5)))))) or ((x5) and ((x15) and (x1)))
    
def pf_v16_c28(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return (not (not ((not ((((((not ((((not ((not ((((((not (((((not (x9)) and (x14)) and (x8)) ^ (x2)) ^ (x5))) and (x13)) or (x1)) and (not (not (x3)))) ^ (x13)) ^ (x5))) ^ ((x15) or (x6)))) and (x11)) and (x9)) or (x10))) and ((x6) ^ (x4))) or (x11)) and (x12)) and ((x2) and (((x3) and (x0)) ^ (x11)))) ^ (((x15) and (x1)) and ((x7) ^ (x8))))) and ((x15) or (((x7) ^ (x8)) ^ (x5)))))) or ((x5) and ((x15) and (x1)))
    

@classical_function
def cf_v16_c29(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return (((not (not ((((((not (((((((((not (((not (((((((x6) ^ (x1)) ^ (x15)) and (x6)) and (x3)) and (x3)) ^ (x2))) and (x13)) ^ ((x13) and (x12)))) and ((x5) and ((x8) and ((x6) ^ (x10))))) and (x4)) or (x4)) ^ (x3)) and (x12)) ^ ((x9) ^ (x12))) and ((x3) and (x0))) or (x1))) and ((x3) ^ (x14))) and (x13)) or ((x15) or (x11))) and (x11)) or ((not ((x9) ^ (x12))) and (x14))))) or ((x7) or (x7))) or ((x6) ^ (x8))) and (x2)
    
def pf_v16_c29(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return (((not (not ((((((not (((((((((not (((not (((((((x6) ^ (x1)) ^ (x15)) and (x6)) and (x3)) and (x3)) ^ (x2))) and (x13)) ^ ((x13) and (x12)))) and ((x5) and ((x8) and ((x6) ^ (x10))))) and (x4)) or (x4)) ^ (x3)) and (x12)) ^ ((x9) ^ (x12))) and ((x3) and (x0))) or (x1))) and ((x3) ^ (x14))) and (x13)) or ((x15) or (x11))) and (x11)) or ((not ((x9) ^ (x12))) and (x14))))) or ((x7) or (x7))) or ((x6) ^ (x8))) and (x2)
    

@classical_function
def cf_v16_c30(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1) -> Int1:
    return not (((((((not (not ((((not (((((((((((((((((not (x10)) and (x9)) ^ (x5)) ^ (not (x9))) and ((x0) and (x15))) and (x5)) ^ (x4)) ^ ((x7) ^ ((x0) and (x15)))) or ((x0) and (x15))) or ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x10)) or (x2)) ^ (x6)) and (x11)) ^ ((x2) and (x4))) or (x14)) and (x14))) or ((x9) and (x8))) ^ ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x8)))) and ((x3) ^ ((x12) ^ (x2)))) and (not ((x12) or (x4)))) or ((not (x9)) and (x8))) and (x2)) or (x13)) and (x13))
    
def pf_v16_c30(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    return not (((((((not (not ((((not (((((((((((((((((not (x10)) and (x9)) ^ (x5)) ^ (not (x9))) and ((x0) and (x15))) and (x5)) ^ (x4)) ^ ((x7) ^ ((x0) and (x15)))) or ((x0) and (x15))) or ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x10)) or (x2)) ^ (x6)) and (x11)) ^ ((x2) and (x4))) or (x14)) and (x14))) or ((x9) and (x8))) ^ ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x8)))) and ((x3) ^ ((x12) ^ (x2)))) and (not ((x12) or (x4)))) or ((not (x9)) and (x8))) and (x2)) or (x13)) and (x13))
    

@classical_function
def cf_v17_c25(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1) -> Int1:
    return (not (((not (not ((((((((((((((((((((x2) ^ (x5)) and (x2)) or (x16)) or ((x15) and (x16))) and (x16)) or ((x12) and (x2))) ^ (x0)) ^ (x9)) or ((x1) or ((x12) and (x2)))) and (((x3) ^ (x6)) ^ ((x9) or (x10)))) or ((x15) and (x16))) and ((x13) ^ (x6))) and (x8)) and (((x9) or (x10)) or (x6))) and ((x4) or (x1))) and (x11)) ^ ((x15) and (x16))) and (x2)) ^ (x14)))) or (x13)) and ((x12) and (x2)))) or ((x2) or (x7))
    
def pf_v17_c25(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    return (not (((not (not ((((((((((((((((((((x2) ^ (x5)) and (x2)) or (x16)) or ((x15) and (x16))) and (x16)) or ((x12) and (x2))) ^ (x0)) ^ (x9)) or ((x1) or ((x12) and (x2)))) and (((x3) ^ (x6)) ^ ((x9) or (x10)))) or ((x15) and (x16))) and ((x13) ^ (x6))) and (x8)) and (((x9) or (x10)) or (x6))) and ((x4) or (x1))) and (x11)) ^ ((x15) and (x16))) and (x2)) ^ (x14)))) or (x13)) and ((x12) and (x2)))) or ((x2) or (x7))
    

@classical_function
def cf_v17_c27(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1) -> Int1:
    return (((((((((((not ((((((((((((((((x0) and (x9)) ^ (x9)) and (x7)) or (x4)) or (x4)) or (x15)) or ((x13) or (x4))) or (x3)) or (x2)) ^ (((x7) and (x14)) or (x14))) or (((x7) and (x14)) or (x8))) or (((x7) and (x14)) or (x8))) ^ (((x7) and (x14)) or (x14))) ^ ((x0) or (x6))) and (x11))) or (x15)) and (((x7) and (x14)) or (x8))) ^ (x10)) or (x6)) ^ ((x7) and (x14))) or (x7)) or ((x5) or (x12))) ^ (x14)) or (x13)) and (((x7) and (x14)) or (x14))) and ((x16) or (x15))
    
def pf_v17_c27(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    return (((((((((((not ((((((((((((((((x0) and (x9)) ^ (x9)) and (x7)) or (x4)) or (x4)) or (x15)) or ((x13) or (x4))) or (x3)) or (x2)) ^ (((x7) and (x14)) or (x14))) or (((x7) and (x14)) or (x8))) or (((x7) and (x14)) or (x8))) ^ (((x7) and (x14)) or (x14))) ^ ((x0) or (x6))) and (x11))) or (x15)) and (((x7) and (x14)) or (x8))) ^ (x10)) or (x6)) ^ ((x7) and (x14))) or (x7)) or ((x5) or (x12))) ^ (x14)) or (x13)) and (((x7) and (x14)) or (x14))) and ((x16) or (x15))
    

@classical_function
def cf_v17_c30(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1) -> Int1:
    return ((((not (((((((not (((((((((((((((((((x6) or (x15)) ^ (x7)) or (x10)) and (x7)) ^ (x14)) ^ (x0)) and (x12)) or (x16)) or (x4)) ^ ((x2) ^ (x5))) or (x4)) or (x6)) and (x11)) or ((x7) or (x14))) or (x15)) ^ ((x8) or (x14))) and ((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13)))) ^ (((x8) or (x14)) and (x0)))) and ((x2) ^ (x5))) or (x15)) or ((x0) and (x14))) ^ (not (((x8) or (x14)) and (x0)))) and ((((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13))) ^ (x3)) or ((x12) or ((x3) and (x2))))) or ((x12) or ((x3) and (x2))))) ^ ((x15) and (not (((x8) or (x14)) and (x0))))) or (((x7) or (x14)) or (x13))) or (x2)) ^ (x9)
    
def pf_v17_c30(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    return ((((not (((((((not (((((((((((((((((((x6) or (x15)) ^ (x7)) or (x10)) and (x7)) ^ (x14)) ^ (x0)) and (x12)) or (x16)) or (x4)) ^ ((x2) ^ (x5))) or (x4)) or (x6)) and (x11)) or ((x7) or (x14))) or (x15)) ^ ((x8) or (x14))) and ((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13)))) ^ (((x8) or (x14)) and (x0)))) and ((x2) ^ (x5))) or (x15)) or ((x0) and (x14))) ^ (not (((x8) or (x14)) and (x0)))) and ((((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13))) ^ (x3)) or ((x12) or ((x3) and (x2))))) or ((x12) or ((x3) and (x2))))) ^ ((x15) and (not (((x8) or (x14)) and (x0))))) or (((x7) or (x14)) or (x13))) or (x2)) ^ (x9)
    

@classical_function
def cf_v17_c31(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1) -> Int1:
    return not (((((((((((not (((not (((((not ((not (((((not (((not ((x2) and (x0))) or (x9)) or (x12))) or (x9)) and (x6)) ^ (x7)) or (x4))) and (x16))) or (not ((not (x1)) and (not (x1))))) and ((x13) or (x1))) ^ (x15)) and (x10))) and ((x3) and (x15))) ^ (((x16) or (x14)) and ((not (x1)) and (not (x1)))))) and ((x1) and ((x0) and (x16)))) and ((x0) and (x16))) or ((x8) or (x6))) and (x11)) ^ (not ((x0) ^ ((x16) or (x14))))) or ((not (not (x10))) ^ ((x0) ^ ((x16) or (x14))))) and (x14)) and (x5)) and ((x8) and (x2))) ^ ((x1) and ((x0) and (x16))))
    
def pf_v17_c31(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    return not (((((((((((not (((not (((((not ((not (((((not (((not ((x2) and (x0))) or (x9)) or (x12))) or (x9)) and (x6)) ^ (x7)) or (x4))) and (x16))) or (not ((not (x1)) and (not (x1))))) and ((x13) or (x1))) ^ (x15)) and (x10))) and ((x3) and (x15))) ^ (((x16) or (x14)) and ((not (x1)) and (not (x1)))))) and ((x1) and ((x0) and (x16)))) and ((x0) and (x16))) or ((x8) or (x6))) and (x11)) ^ (not ((x0) ^ ((x16) or (x14))))) or ((not (not (x10))) ^ ((x0) ^ ((x16) or (x14))))) and (x14)) and (x5)) and ((x8) and (x2))) ^ ((x1) and ((x0) and (x16))))
    

@classical_function
def cf_v18_c30(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1, x17 : Int1) -> Int1:
    return ((((((((((((((not (((not (((not ((not ((((((not ((x4) or (x14))) and (x3)) ^ (x3)) ^ (x13)) or (x3)) ^ (x10))) ^ ((x17) or (x11)))) and (x17)) ^ (x9))) or (((x11) ^ (x15)) ^ (x15))) and ((x6) ^ (x12)))) or (x11)) ^ (x16)) and (not (x17))) ^ (x3)) or (x8)) or (x5)) or ((x15) or (((x11) ^ (x15)) ^ (x15)))) ^ ((x14) ^ (x8))) or (x0)) and ((x11) ^ (x15))) or ((x4) ^ (x7))) or (x2)) or (not (x9))) or (((x14) ^ (x8)) or ((x11) ^ (x15)))
    
def pf_v18_c30(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
    return ((((((((((((((not (((not (((not ((not ((((((not ((x4) or (x14))) and (x3)) ^ (x3)) ^ (x13)) or (x3)) ^ (x10))) ^ ((x17) or (x11)))) and (x17)) ^ (x9))) or (((x11) ^ (x15)) ^ (x15))) and ((x6) ^ (x12)))) or (x11)) ^ (x16)) and (not (x17))) ^ (x3)) or (x8)) or (x5)) or ((x15) or (((x11) ^ (x15)) ^ (x15)))) ^ ((x14) ^ (x8))) or (x0)) and ((x11) ^ (x15))) or ((x4) ^ (x7))) or (x2)) or (not (x9))) or (((x14) ^ (x8)) or ((x11) ^ (x15)))
    

@classical_function
def cf_v18_c31(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1, x17 : Int1) -> Int1:
    return (((not (not (not ((not (((((((not (((((not ((not ((((((((((x14) or (x16)) and (x4)) and (x11)) or (x9)) ^ (x4)) ^ (x2)) or ((x9) or (x13))) and (x3)) ^ ((x12) or (x3)))) and (x15))) ^ (x5)) ^ (((x1) ^ (x5)) ^ ((x14) ^ (x15)))) ^ ((x1) ^ (x5))) and (x8))) and (x10)) or (x7)) or (((x12) and (x16)) and (x17))) and (((x13) or (x5)) or ((x7) or (x0)))) and ((x6) and (x16))) or (x6))) and ((x9) or (x13)))))) and (x13)) or (not ((x10) and ((x14) ^ (x15))))) or (x3)
    
def pf_v18_c31(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
    return (((not (not (not ((not (((((((not (((((not ((not ((((((((((x14) or (x16)) and (x4)) and (x11)) or (x9)) ^ (x4)) ^ (x2)) or ((x9) or (x13))) and (x3)) ^ ((x12) or (x3)))) and (x15))) ^ (x5)) ^ (((x1) ^ (x5)) ^ ((x14) ^ (x15)))) ^ ((x1) ^ (x5))) and (x8))) and (x10)) or (x7)) or (((x12) and (x16)) and (x17))) and (((x13) or (x5)) or ((x7) or (x0)))) and ((x6) and (x16))) or (x6))) and ((x9) or (x13)))))) and (x13)) or (not ((x10) and ((x14) ^ (x15))))) or (x3)
    

@classical_function
def cf_v19_c32(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1, x17 : Int1, x18 : Int1) -> Int1:
    return ((((not ((not (((((((((((((((((((not ((not ((not ((not (x16)) and (x2))) or ((x0) and (x13)))) or (x2))) ^ (x18)) and (x6)) or (((x0) and (x13)) ^ (x9))) ^ ((x0) and (x13))) ^ (x0)) ^ (x4)) and (not ((x0) or (x1)))) and (x11)) or ((not (not (x7))) and (x5))) or (x12)) and (((x0) or (x1)) or (x8))) and (x4)) or (not (x7))) or (((x0) or (x1)) or (x8))) and (x17)) ^ ((((x0) or (x1)) or (x8)) ^ (x3))) ^ (x15)) and ((x8) or (x6)))) ^ ((x16) and (x10)))) ^ (x14)) or (((x0) and (x13)) or (x6))) ^ (x0)) ^ ((not (not (x7))) and (x5))
    
def pf_v19_c32(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18):
    return ((((not ((not (((((((((((((((((((not ((not ((not ((not (x16)) and (x2))) or ((x0) and (x13)))) or (x2))) ^ (x18)) and (x6)) or (((x0) and (x13)) ^ (x9))) ^ ((x0) and (x13))) ^ (x0)) ^ (x4)) and (not ((x0) or (x1)))) and (x11)) or ((not (not (x7))) and (x5))) or (x12)) and (((x0) or (x1)) or (x8))) and (x4)) or (not (x7))) or (((x0) or (x1)) or (x8))) and (x17)) ^ ((((x0) or (x1)) or (x8)) ^ (x3))) ^ (x15)) and ((x8) or (x6)))) ^ ((x16) and (x10)))) ^ (x14)) or (((x0) and (x13)) or (x6))) ^ (x0)) ^ ((not (not (x7))) and (x5))
    

@classical_function
def cf_v19_c33(x0 : Int1, x1 : Int1, x2 : Int1, x3 : Int1, x4 : Int1, x5 : Int1, x6 : Int1, x7 : Int1, x8 : Int1, x9 : Int1, x10 : Int1, x11 : Int1, x12 : Int1, x13 : Int1, x14 : Int1, x15 : Int1, x16 : Int1, x17 : Int1, x18 : Int1) -> Int1:
    return (((((not ((not ((((((((((((((((((((((not (not (not (not (x0))))) and (not (x10))) or ((x16) and (not (x10)))) ^ (x12)) or (x17)) or (x12)) or (x8)) and (x9)) and (x3)) ^ (x14)) or ((x16) and (not (x10)))) and ((x7) or (x16))) or ((x1) and (x15))) ^ (((x10) or (not (x10))) and (x15))) or (x6)) and (x2)) or ((x17) or (x11))) or (x13)) ^ (x4)) ^ (((x10) or (not (x10))) and (x15))) or ((x18) or (x13))) and ((x16) and (not (x10))))) ^ (((((x17) or (x11)) ^ (x7)) or (not (x5))) or (((x1) and (x15)) ^ (x15))))) ^ (x14)) and (x9)) or (not (x5))) or ((x18) or (x13))) or (not (x5))
    
def pf_v19_c33(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18):
    return (((((not ((not ((((((((((((((((((((((not (not (not (not (x0))))) and (not (x10))) or ((x16) and (not (x10)))) ^ (x12)) or (x17)) or (x12)) or (x8)) and (x9)) and (x3)) ^ (x14)) or ((x16) and (not (x10)))) and ((x7) or (x16))) or ((x1) and (x15))) ^ (((x10) or (not (x10))) and (x15))) or (x6)) and (x2)) or ((x17) or (x11))) or (x13)) ^ (x4)) ^ (((x10) or (not (x10))) and (x15))) or ((x18) or (x13))) and ((x16) and (not (x10))))) ^ (((((x17) or (x11)) ^ (x7)) or (not (x5))) or (((x1) and (x15)) ^ (x15))))) ^ (x14)) and (x9)) or (not (x5))) or ((x18) or (x13))) or (not (x5))
    
benchmark_functions = {'(2,1)': {'variables': ['x0', 'x1'], 'statement': '(x1) ^ (x0)', 'classical_function': 'cf_v2_c1', 'python_function': 'pf_v2_c1'}, '(2,2)': {'variables': ['x0', 'x1'], 'statement': '((x1) and (x0)) and (x0)', 'classical_function': 'cf_v2_c2', 'python_function': 'pf_v2_c2'}, '(2,3)': {'variables': ['x0', 'x1'], 'statement': 'not (((x1) ^ (x0)) or ((x0) ^ (x0)))', 'classical_function': 'cf_v2_c3', 'python_function': 'pf_v2_c3'}, '(2,4)': {'variables': ['x0', 'x1'], 'statement': '((not (not (x1))) or (not (x0))) ^ ((not (x1)) or (not (x1)))', 'classical_function': 'cf_v2_c4', 'python_function': 'pf_v2_c4'}, '(2,5)': {'variables': ['x0', 'x1'], 'statement': '((not (((x1) and (x1)) and (x1))) and ((x0) and (x1))) ^ (not (((x0) and (x1)) and ((x0) and (x1))))', 'classical_function': 'cf_v2_c5', 'python_function': 'pf_v2_c5'}, '(2,6)': {'variables': ['x0', 'x1'], 'statement': '((((((x0) and (x1)) or (x1)) ^ (((x1) and (x0)) or ((x1) and (x0)))) ^ (x0)) or (((x1) and (x0)) ^ (((x1) and (x0)) or ((x1) and (x0))))) or ((x1) and (x0))', 'classical_function': 'cf_v2_c6', 'python_function': 'pf_v2_c6'}, '(2,7)': {'variables': ['x0', 'x1'], 'statement': 'not ((((((not (x1)) or (not (x1))) and ((x0) or (not (x1)))) and (x0)) ^ (x0)) and (((x1) and ((x0) or (not (x1)))) ^ (not (x1))))', 'classical_function': 'cf_v2_c7', 'python_function': 'pf_v2_c7'}, '(2,8)': {'variables': ['x0', 'x1'], 'statement': '((not (not (((not ((x0) ^ (x1))) ^ (not ((x1) ^ (x1)))) and ((x1) ^ (x1))))) ^ (x1)) and (not ((x1) ^ (x1)))', 'classical_function': 'cf_v2_c8', 'python_function': 'pf_v2_c8'}, '(2,9)': {'variables': ['x0', 'x1'], 'statement': 'not (not (not (((((((x1) or (x1)) ^ ((x0) or (x0))) ^ (x1)) ^ (x1)) and ((x1) ^ ((x0) or (x0)))) and ((x0) and (x1)))))', 'classical_function': 'cf_v2_c9', 'python_function': 'pf_v2_c9'}, '(2,10)': {'variables': ['x0', 'x1'], 'statement': '(((not ((((not (not ((x1) and (x1)))) or (x0)) ^ ((x0) or (x0))) and (x0))) and ((x0) or (x0))) or (x0)) and ((not ((x0) and (x1))) and (not (x1)))', 'classical_function': 'cf_v2_c10', 'python_function': 'pf_v2_c10'}, '(2,11)': {'variables': ['x0', 'x1'], 'statement': 'not (not ((((((((((x1) or (x0)) or ((x1) or (x1))) and (((x1) or (x1)) or (x0))) or ((x1) or (x1))) or ((x1) or (x1))) ^ ((x1) or (x0))) ^ ((((x1) or (x1)) and (((x1) or (x1)) or (x0))) or ((x1) or (x0)))) and ((x1) or (x0))) or (((x1) or (x1)) and (((x1) or (x1)) or (x0)))))', 'classical_function': 'cf_v2_c11', 'python_function': 'pf_v2_c11'}, '(2,12)': {'variables': ['x0', 'x1'], 'statement': '(not (((((not (((((not (x1)) and (x1)) ^ (x1)) and (not (x0))) ^ (x0))) or ((x0) and (x1))) and (x1)) or ((x0) and (x0))) ^ ((not (not (x0))) and (not (not (x0)))))) ^ (x1)', 'classical_function': 'cf_v2_c12', 'python_function': 'pf_v2_c12'}, '(2,13)': {'variables': ['x0', 'x1'], 'statement': '(not ((((((not (((((not (x0)) and (not (x1))) and (x1)) or ((x0) and (x1))) or (not (x1)))) ^ ((x0) and (x1))) or ((not (x1)) and (x0))) and (not (x1))) and ((x0) or (x1))) or (not (x1)))) ^ (not (x1))', 'classical_function': 'cf_v2_c13', 'python_function': 'pf_v2_c13'}, '(2,14)': {'variables': ['x0', 'x1'], 'statement': '((not ((((((not ((((((x1) ^ (x1)) ^ (x0)) and ((x1) ^ (x1))) and ((x1) ^ (x1))) and (((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))))) or ((((x1) ^ (x1)) and (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or (((x1) ^ (x0)) and ((x1) ^ (x1)))) and (((x1) ^ (x0)) or (x0))) ^ ((x1) and (((x1) ^ (x0)) or (x0)))) and (((x1) ^ (x0)) and ((x1) ^ (x1))))) or ((x1) and (((x1) ^ (x0)) or (x0)))) ^ (not (x0))', 'classical_function': 'cf_v2_c14', 'python_function': 'pf_v2_c14'}, '(2,15)': {'variables': ['x0', 'x1'], 'statement': 'not (((((not (not ((((((not (not (not (x0)))) ^ (not (x0))) and (not (x0))) and (x1)) or (x1)) and (x0)))) and ((x0) and (not (x0)))) ^ ((((not (x0)) ^ (not (x0))) or (x0)) and ((not (x0)) and (((not (x0)) ^ (not (x0))) or (x0))))) and (((not (x0)) ^ (not (x0))) or (x0))) ^ ((not (x0)) and (not ((x0) and (not (x0))))))', 'classical_function': 'cf_v2_c15', 'python_function': 'pf_v2_c15'}, '(2,16)': {'variables': ['x0', 'x1'], 'statement': 'not (not (((not ((not (not ((not (((((not ((x1) or (x0))) or ((x0) or (x0))) and ((x0) or (x0))) and (not ((x0) or (x0)))) and (((x0) or (x0)) and (not ((x0) or (x0)))))) and ((x1) and (not ((x0) or (x0))))))) and ((x0) or ((x0) or (x0))))) or (((x0) or ((x0) or (x0))) and ((x0) or (x0)))) or (((x0) or (x0)) and (not ((x0) or (x0))))))', 'classical_function': 'cf_v2_c16', 'python_function': 'pf_v2_c16'}, '(3,2)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((x0) ^ (x1)) and (x2)', 'classical_function': 'cf_v3_c2', 'python_function': 'pf_v3_c2'}, '(3,3)': {'variables': ['x0', 'x1', 'x2'], 'statement': '(not ((x1) and (x0))) or ((x1) and (x2))', 'classical_function': 'cf_v3_c3', 'python_function': 'pf_v3_c3'}, '(3,4)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not ((((x1) and (x1)) and (x2)) or (x0))', 'classical_function': 'cf_v3_c4', 'python_function': 'pf_v3_c4'}, '(3,5)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((((not (x0)) ^ (x0)) or ((x2) ^ (x0))) or (x2)) and (((x2) ^ (x0)) or (x1))', 'classical_function': 'cf_v3_c5', 'python_function': 'pf_v3_c5'}, '(3,6)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((((((x0) ^ (x1)) or (x2)) and (x2)) or (x2)) and (x1)) and ((x2) or ((x0) ^ (x1)))', 'classical_function': 'cf_v3_c6', 'python_function': 'pf_v3_c6'}, '(3,7)': {'variables': ['x0', 'x1', 'x2'], 'statement': '(not (not (((((x2) or (x2)) ^ (x2)) or ((x2) ^ ((x2) or (x0)))) or ((x1) or (x1))))) and (not (x0))', 'classical_function': 'cf_v3_c7', 'python_function': 'pf_v3_c7'}, '(3,8)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((((((not ((x0) ^ (x0))) or (x2)) ^ ((not (x1)) or (x0))) ^ ((x2) ^ (x1))) ^ ((x2) ^ ((x2) ^ (x1)))) or ((x2) ^ (x1))) and ((not (x1)) or (x2))', 'classical_function': 'cf_v3_c8', 'python_function': 'pf_v3_c8'}, '(3,9)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not (((not ((((not (not (x0))) or (not (not (x1)))) or (not (x1))) ^ (x0))) ^ (not ((x2) or (x2)))) ^ (not (not (x1))))', 'classical_function': 'cf_v3_c9', 'python_function': 'pf_v3_c9'}, '(3,10)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not (((not (((((((x0) and (x1)) or ((x0) and (x2))) ^ (x0)) and (((x2) or (x0)) ^ (x0))) ^ ((x0) and (x2))) ^ ((x0) and (x2)))) and ((x2) or (x0))) or (x0))', 'classical_function': 'cf_v3_c10', 'python_function': 'pf_v3_c10'}, '(3,11)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((((not ((((((not (x1)) or (not (x0))) ^ (not (x0))) or (x0)) or ((x2) or (x2))) ^ ((x2) or (x2)))) and (not ((x2) ^ (x1)))) ^ (x1)) ^ (x1)) or ((x0) ^ (x2))', 'classical_function': 'cf_v3_c11', 'python_function': 'pf_v3_c11'}, '(3,12)': {'variables': ['x0', 'x1', 'x2'], 'statement': '(not (not (((((not (((not ((x1) and (x2))) and (not (x0))) or (x1))) ^ (x0)) ^ ((not (x1)) ^ (x2))) or (x2)) ^ ((x2) and (not (x0)))))) ^ ((not (x0)) or (x1))', 'classical_function': 'cf_v3_c12', 'python_function': 'pf_v3_c12'}, '(3,13)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not (not ((((((not ((((((x2) ^ (x0)) or (x1)) or (x0)) ^ ((x0) ^ (x2))) and ((x2) or (x0)))) ^ (((x2) or (x0)) or (x2))) ^ (x1)) or (not ((x0) ^ (x2)))) ^ (not ((x0) ^ (x2)))) ^ (x2)))', 'classical_function': 'cf_v3_c13', 'python_function': 'pf_v3_c13'}, '(3,14)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not ((not (not (((not (((not (((((x0) or (x2)) ^ (x2)) ^ (x0)) and (x1))) ^ ((x0) ^ (x1))) ^ (x0))) and (not ((x2) or (x0)))) or (x0)))) ^ (x1))', 'classical_function': 'cf_v3_c14', 'python_function': 'pf_v3_c14'}, '(3,15)': {'variables': ['x0', 'x1', 'x2'], 'statement': '(not ((not ((((not ((((((not ((x1) ^ (x1))) ^ (x0)) or (not (x0))) or (x1)) and (x2)) or (x0))) ^ (x0)) ^ (x2)) ^ ((x0) ^ (not (x0))))) ^ ((x1) ^ (x1)))) and ((x0) or ((x1) or (not (x0))))', 'classical_function': 'cf_v3_c15', 'python_function': 'pf_v3_c15'}, '(3,16)': {'variables': ['x0', 'x1', 'x2'], 'statement': '((not ((((((((((not ((not (not (x1))) ^ (x2))) ^ ((not (x0)) ^ (not (not (x0))))) and (not (x0))) ^ (x0)) or (not (not (x0)))) and (not (x2))) ^ (not (x2))) or (((not (x0)) ^ (not (not (x0)))) ^ (x0))) ^ ((x0) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((((not (x0)) ^ (not (not (x0)))) ^ (x0)) or (not (x0))))) or ((x0) ^ (x1))) ^ (((not (x0)) ^ (not (not (x0)))) ^ (x0))', 'classical_function': 'cf_v3_c16', 'python_function': 'pf_v3_c16'}, '(3,17)': {'variables': ['x0', 'x1', 'x2'], 'statement': 'not ((((((not (((((not (((not ((not (x0)) ^ (x0))) or (x1)) or (x0))) ^ (x1)) and ((x2) ^ (x1))) and ((x2) or (x0))) ^ ((not (not (x1))) or (x2)))) and (not (not (x1)))) ^ (x2)) and ((x2) or (x0))) or (not (x1))) ^ ((x0) and ((x2) ^ (x1))))', 'classical_function': 'cf_v3_c17', 'python_function': 'pf_v3_c17'}, '(4,3)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(not ((x3) ^ (x0))) ^ (not ((x1) ^ (x2)))', 'classical_function': 'cf_v4_c3', 'python_function': 'pf_v4_c3'}, '(4,4)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(not (((x0) or (x3)) and ((x1) or (x0)))) ^ (x2)', 'classical_function': 'cf_v4_c4', 'python_function': 'pf_v4_c4'}, '(4,5)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '((((not (x1)) and (x2)) ^ (x2)) ^ (x3)) and (x0)', 'classical_function': 'cf_v4_c5', 'python_function': 'pf_v4_c5'}, '(4,6)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '((((((x3) or (x1)) and ((x3) or (x2))) or ((x0) and (x2))) ^ (x2)) or ((x3) or (x2))) ^ (x1)', 'classical_function': 'cf_v4_c6', 'python_function': 'pf_v4_c6'}, '(4,7)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(((((not ((x0) and (x0))) ^ (x2)) ^ (((x2) and (x3)) ^ ((x2) and (x3)))) and ((x2) and (x3))) or (x1)) and (x2)', 'classical_function': 'cf_v4_c7', 'python_function': 'pf_v4_c7'}, '(4,8)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': 'not (((((((not (x2)) and (x0)) and (x0)) ^ ((not (x1)) and (x3))) ^ (x2)) ^ (x1)) and (x0))', 'classical_function': 'cf_v4_c8', 'python_function': 'pf_v4_c8'}, '(4,9)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '((((not (((((x3) and (x2)) or ((x3) and (x0))) ^ (x1)) ^ (x0))) and (x1)) ^ (((x3) and (x0)) ^ ((x3) and (x0)))) or (x1)) and (x0)', 'classical_function': 'cf_v4_c9', 'python_function': 'pf_v4_c9'}, '(4,10)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': 'not (((((not (((((x0) ^ (x2)) ^ (x2)) or (x3)) and ((x3) ^ (x0)))) and ((x1) or (x3))) and (x3)) and (not ((x1) or (x3)))) ^ (((x2) ^ (x1)) and ((x3) ^ (x0))))', 'classical_function': 'cf_v4_c10', 'python_function': 'pf_v4_c10'}, '(4,11)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '((((not (((((((x1) ^ (x1)) and (x3)) ^ (x3)) and (x2)) and ((x0) and ((x0) ^ (x1)))) ^ (x0))) or ((x0) ^ (x1))) ^ (not (((x1) ^ (x1)) ^ ((x1) and (x0))))) ^ ((x1) and (x0))) or (((x1) and (x0)) and ((x0) and ((x0) ^ (x1))))', 'classical_function': 'cf_v4_c11', 'python_function': 'pf_v4_c11'}, '(4,12)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(((not (((((((((x0) ^ (x3)) ^ (x1)) ^ (x3)) or (((x3) ^ (x0)) ^ (x1))) and (((x3) ^ (x0)) ^ (x1))) ^ (x1)) and ((x0) and (x2))) or ((x1) ^ (((x3) ^ (x0)) ^ (x1))))) or (x2)) and ((x1) ^ (((x3) ^ (x0)) ^ (x1)))) ^ ((x0) and (x2))', 'classical_function': 'cf_v4_c12', 'python_function': 'pf_v4_c12'}, '(4,13)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '((((((not (not ((((((x0) or (x2)) and (x2)) or (x2)) and (x2)) or ((x2) or (x2))))) or (x3)) or (not (x1))) ^ (not ((x1) and (x2)))) or ((x0) or (((x2) or (x2)) or (x3)))) ^ ((x0) or (x3))) ^ ((x0) or (x3))', 'classical_function': 'cf_v4_c13', 'python_function': 'pf_v4_c13'}, '(4,14)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(not ((((((((((((not (x1)) or (not (x3))) ^ (x0)) and (((x3) or (x0)) ^ (x2))) or (not (x3))) and (x2)) and (x3)) and ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) or (x1)) or (((x3) or (x0)) and (x2))) ^ ((((x3) or (x0)) ^ (x2)) and ((x3) or (x0)))) and ((x0) or ((((x3) or (x0)) ^ (x2)) or ((x2) and ((x3) or (x0))))))) or (x2)', 'classical_function': 'cf_v4_c14', 'python_function': 'pf_v4_c14'}, '(4,15)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(((not ((not ((((not ((not ((((x0) ^ (x1)) and ((x3) ^ (x0))) ^ (x2))) and ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (not ((x3) ^ (x0)))) or (x0))) and (x3))) and (x3)) or (not (x0))) or (x0)', 'classical_function': 'cf_v4_c15', 'python_function': 'pf_v4_c15'}, '(4,16)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(((((((not ((not (not (((not (((x3) ^ (x0)) and (x0))) or (x1)) and ((x2) and (x1))))) and (((x1) ^ (x3)) or ((x1) ^ (x3))))) and (((x1) ^ (x3)) or ((x1) ^ (x3)))) and ((not ((x1) ^ (x3))) and ((x2) and (x1)))) or ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not (x1))) ^ ((not ((x1) ^ (x3))) and ((x2) and (x1)))) ^ (not ((x1) ^ (x3)))) or (((x1) ^ (x3)) or ((x1) ^ (x3)))', 'classical_function': 'cf_v4_c16', 'python_function': 'pf_v4_c16'}, '(4,17)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(((((((not (((((((((not (x1)) and (not (x1))) or (x2)) and ((x2) or (x1))) or (x3)) ^ ((x0) and (x2))) and ((x1) and (x3))) or (x3)) ^ (x3))) or ((not (x1)) and ((x2) or (x1)))) and (x0)) or (x3)) or ((not (x1)) and ((x2) or (x1)))) ^ (((x3) ^ ((x1) and (x3))) or (not (x1)))) or ((x1) and (x3))) ^ (x3)', 'classical_function': 'cf_v4_c17', 'python_function': 'pf_v4_c17'}, '(4,18)': {'variables': ['x0', 'x1', 'x2', 'x3'], 'statement': '(not (((((not (((not (((not ((((not (not (x0))) and (x3)) ^ (not (x0))) ^ (x3))) and (x0)) ^ (not (not (x0))))) and ((not (x0)) ^ (x0))) or (x1))) ^ (not (x0))) and ((x2) and (x3))) and ((not (x0)) ^ (not (x0)))) ^ ((x2) and (x3)))) and (x3)', 'classical_function': 'cf_v4_c18', 'python_function': 'pf_v4_c18'}, '(5,4)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((((x3) and (x1)) and (x0)) ^ ((x1) and ((x4) and (x1)))) and ((x2) ^ (x2))', 'classical_function': 'cf_v5_c4', 'python_function': 'pf_v5_c4'}, '(5,5)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((((x2) ^ (x0)) or (x0)) ^ (x4)) ^ ((x1) ^ (x3))) ^ (((x1) ^ (x3)) or (x0))', 'classical_function': 'cf_v5_c5', 'python_function': 'pf_v5_c5'}, '(5,6)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((((((x3) ^ (x2)) or (x0)) or (x3)) ^ ((x3) or (x3))) ^ ((x4) ^ ((x3) or (x3)))) and (x1)', 'classical_function': 'cf_v5_c6', 'python_function': 'pf_v5_c6'}, '(5,7)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((not (((not ((x3) and (x3))) ^ (not (x2))) and ((x4) and (x2)))) and (x1)) and (x0)', 'classical_function': 'cf_v5_c7', 'python_function': 'pf_v5_c7'}, '(5,8)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': 'not ((((((((x4) or (x4)) ^ (x2)) or (x3)) and (x2)) or ((x4) and (x0))) or (((x0) ^ (x2)) or (x4))) or (((x2) or (x1)) or (x0)))', 'classical_function': 'cf_v5_c8', 'python_function': 'pf_v5_c8'}, '(5,9)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((((((not ((x4) or (x3))) ^ (x0)) and (x2)) ^ (not (x1))) ^ (x2)) and (x4)) or (x4)) ^ (((x0) ^ (x0)) and (x2))', 'classical_function': 'cf_v5_c9', 'python_function': 'pf_v5_c9'}, '(5,10)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((((((not ((not ((x3) ^ (x4))) or (not (x0)))) or (x2)) and ((x1) ^ (x4))) ^ (x2)) and (x1)) ^ ((x0) or (x4))) or (not (x0))', 'classical_function': 'cf_v5_c10', 'python_function': 'pf_v5_c10'}, '(5,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((((not (((not (((x0) and (x1)) or (x4))) and (x0)) and ((x2) or ((x2) and (x3))))) or (x4)) ^ ((x1) or (x4))) and ((x2) and (x3))) and (((x1) and (x2)) ^ (x2))) and ((x2) or ((x2) and (x3)))', 'classical_function': 'cf_v5_c11', 'python_function': 'pf_v5_c11'}, '(5,12)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((((((((not (((x3) and (x1)) or (x3))) ^ ((x2) and (x3))) ^ (((x2) and (x3)) ^ (x1))) or (x4)) and (((x2) and (x3)) ^ (x1))) or (x1)) and ((x1) or (x0))) ^ (((x1) or (x2)) and (x0))) or (x1)) or (x0)', 'classical_function': 'cf_v5_c12', 'python_function': 'pf_v5_c12'}, '(5,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((((((((((not (((x2) ^ (x2)) ^ (x1))) ^ (x3)) or ((x4) ^ (x0))) ^ (x0)) and (x2)) and (x1)) or ((x0) and ((x4) ^ (x0)))) ^ ((x2) and (x2))) and ((x2) or (x1))) ^ (((x2) or (x1)) ^ (((x4) ^ (x0)) ^ ((x4) ^ (x0))))) and (x2)', 'classical_function': 'cf_v5_c13', 'python_function': 'pf_v5_c13'}, '(5,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': 'not (((((not ((((((((not (x2)) or (not (x0))) and (x2)) and (not (x0))) or (x4)) or ((x1) or (x0))) and (x0)) and (x4))) or ((x2) and ((x1) or (x0)))) ^ (((x2) and ((x1) or (x0))) or ((x3) and (not (x0))))) ^ (x0)) and (x3))', 'classical_function': 'cf_v5_c14', 'python_function': 'pf_v5_c14'}, '(5,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((not (not (((not (not ((((((not (x0)) ^ (x2)) or (x4)) or ((x4) ^ (x3))) or (x4)) or ((x1) or (x2))))) or (not ((x3) or ((x1) or (x2))))) ^ ((x1) or (x2))))) and (not ((not (x0)) or (x4)))) and (((x4) ^ (x3)) and (x2))) ^ ((x4) or ((x4) ^ (x3)))', 'classical_function': 'cf_v5_c15', 'python_function': 'pf_v5_c15'}, '(5,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((not (((not ((((((((not (((x1) and (x4)) and (x2))) ^ (not ((x1) and (x0)))) ^ (x3)) ^ ((x0) and (x0))) ^ (x0)) or ((x3) ^ (x3))) ^ ((x0) and (x0))) or (x0))) ^ (x4)) or ((not ((x1) and (x0))) ^ (x0)))) ^ (x3)) and (x3)', 'classical_function': 'cf_v5_c16', 'python_function': 'pf_v5_c16'}, '(5,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': 'not ((((((((not ((((((not (((x2) and (x0)) ^ ((x1) and (x4)))) and (x0)) or ((x1) and (x4))) and (x2)) ^ (x0)) and (not (x0)))) or ((x0) ^ (x3))) or ((not (x0)) and (x4))) ^ ((x4) ^ (x3))) and (x3)) or ((x0) ^ (x3))) or (x4)) and ((x0) and (((not (x0)) or (x0)) and ((x1) and (x4)))))', 'classical_function': 'cf_v5_c17', 'python_function': 'pf_v5_c17'}, '(5,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '(((not ((not (not (not ((not ((((((not ((not (x4)) or (x4))) ^ (x1)) ^ (x3)) ^ (x2)) ^ (not (x4))) and (x2))) and (((x4) or (x4)) ^ (x1)))))) ^ (not (x2)))) and (not ((((x4) or (x4)) ^ (x1)) and (x0)))) or ((x4) or (x4))) or ((not (x2)) and (not (x4)))', 'classical_function': 'cf_v5_c18', 'python_function': 'pf_v5_c18'}, '(5,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4'], 'statement': '((((not ((((not ((((((((not (not ((x3) and (x2)))) and (x2)) and (not (x2))) and (x3)) or ((x4) and (not (x4)))) and (not (x4))) ^ (x0)) and ((x0) and (x1)))) and (((x0) and (x1)) and ((x4) and (not (x4))))) and ((x0) and (x1))) or ((not (x2)) and (x1)))) and (x3)) or (((x0) and (x1)) and ((x4) and (not (x4))))) and (((x4) and (not (x4))) ^ ((x3) and (x1)))) or ((x3) and (x1))', 'classical_function': 'cf_v5_c19', 'python_function': 'pf_v5_c19'}, '(6,6)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '((((((x4) ^ (x0)) or (x1)) ^ (x5)) ^ (x4)) ^ ((x1) or ((x2) ^ (x3)))) and (x5)', 'classical_function': 'cf_v6_c6', 'python_function': 'pf_v6_c6'}, '(6,7)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(not ((((((x2) and (x0)) and (x1)) or (x4)) or ((x0) or ((x3) and (x5)))) ^ (x3))) ^ ((x0) or ((x3) and (x5)))', 'classical_function': 'cf_v6_c7', 'python_function': 'pf_v6_c7'}, '(6,8)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': 'not (((((not (((x3) ^ (x1)) and (x0))) and (x1)) and (x4)) or ((x1) and (x5))) and (x2))', 'classical_function': 'cf_v6_c8', 'python_function': 'pf_v6_c8'}, '(6,9)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(((((((not ((x3) or (x5))) or (not (x2))) ^ (x1)) ^ (x3)) or ((x4) ^ ((x0) or (x4)))) or (x0)) and (not (x2))) ^ (x2)', 'classical_function': 'cf_v6_c9', 'python_function': 'pf_v6_c9'}, '(6,10)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '((((((((((x1) ^ (x0)) or (x1)) or (x0)) and (x5)) and (x3)) ^ (x4)) or (x4)) ^ ((x5) ^ (x1))) and ((x4) or (x2))) and (x0)', 'classical_function': 'cf_v6_c10', 'python_function': 'pf_v6_c10'}, '(6,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(not (((((not (not ((((x5) or (x1)) ^ (x2)) and (x1)))) or (((x3) ^ ((x4) or (x4))) and (x4))) and (x0)) and ((((x3) ^ ((x4) or (x4))) and (x4)) or (x0))) or (((x4) and (x0)) and (x3)))) and (not (x2))', 'classical_function': 'cf_v6_c11', 'python_function': 'pf_v6_c11'}, '(6,12)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(((not (((((not ((((x2) ^ (x5)) or (x1)) or (x5))) or (x2)) or (x4)) ^ (not (x5))) and (x4))) or (((x0) ^ (x5)) or (x4))) or (x3)) ^ (((x0) ^ (x5)) or (x4))', 'classical_function': 'cf_v6_c12', 'python_function': 'pf_v6_c12'}, '(6,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': 'not (((((((((((((x5) and (x1)) ^ (x0)) and (x2)) or ((x0) and (x1))) ^ (x0)) or (x2)) ^ ((x0) and (x1))) ^ ((x3) or (x1))) and ((x0) and (x1))) ^ (x4)) and ((x4) and (((x0) and (x1)) and (x2)))) or ((x4) and (((x0) and (x1)) and (x2))))', 'classical_function': 'cf_v6_c13', 'python_function': 'pf_v6_c13'}, '(6,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(((((not ((not (not (not (((not ((x2) or (x2))) ^ ((x2) or (x4))) or (x5))))) ^ (not (x0)))) ^ (not (not ((x5) ^ (not ((x2) or (x4))))))) and (x3)) and (not (x0))) or ((x0) or (x1))) ^ (x0)', 'classical_function': 'cf_v6_c14', 'python_function': 'pf_v6_c14'}, '(6,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(((((((not (((not (((((x3) and (x3)) or (x2)) and (x5)) or (x1))) or (x3)) and (x4))) or (x1)) and (not (x2))) ^ (x0)) or ((x5) and (x4))) or (not (x2))) and ((x2) and (x4))) or (((x1) ^ ((x2) and (x4))) and (not ((x4) and (((x2) and (x4)) or (x0)))))', 'classical_function': 'cf_v6_c15', 'python_function': 'pf_v6_c15'}, '(6,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(not (((((((not (((((not (not ((x4) and (x3)))) or ((x1) and (x1))) ^ (x2)) and (x3)) and (not (x0)))) and (x4)) ^ (not (x0))) and (((x2) ^ (x3)) and ((x1) and (x1)))) ^ (((not (x2)) or (not (x2))) ^ (x0))) ^ (x5)) or (x0))) and (x5)', 'classical_function': 'cf_v6_c16', 'python_function': 'pf_v6_c16'}, '(6,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '((((((not (((((not ((not ((((x2) ^ (x0)) ^ (x1)) ^ (x2))) or ((x5) ^ (x3)))) and (x0)) and (x4)) or ((x0) ^ (x5))) ^ (x3))) ^ (x0)) or (x0)) ^ (not (x2))) ^ (((x5) and (not (x1))) and (x5))) or (x2)) or ((((x5) and (not (x1))) and (x5)) ^ (x0))', 'classical_function': 'cf_v6_c17', 'python_function': 'pf_v6_c17'}, '(6,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(not (((((not (((((((not (((not ((x1) ^ (x0))) or (x0)) or (x2))) or (x0)) and ((x4) ^ (x4))) and (x0)) or (not (x3))) or (((not (not (x3))) and (x3)) or (x1))) and ((x4) ^ (x4)))) or ((x1) and (x5))) or (((x1) and (((x4) ^ (x4)) or (x5))) or (((x4) ^ (x4)) or (x5)))) and (x1)) or (x2))) ^ (x5)', 'classical_function': 'cf_v6_c18', 'python_function': 'pf_v6_c18'}, '(6,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': 'not (not (not ((((((not ((((not ((not ((not (((x0) ^ (x0)) ^ ((x3) ^ (x2)))) ^ (x0))) ^ ((x1) ^ ((x3) ^ (x2))))) or (x1)) or (x2)) ^ (x5))) ^ (x4)) ^ ((x3) ^ (x2))) and ((not ((x1) ^ ((x3) ^ (x2)))) or (not (not ((x1) ^ ((x3) ^ (x2))))))) ^ (not (x4))) ^ ((x2) ^ (not (x4))))))', 'classical_function': 'cf_v6_c19', 'python_function': 'pf_v6_c19'}, '(6,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], 'statement': '(not ((((((not ((((not ((not (not ((((((x0) ^ (x5)) and (x5)) or ((x5) ^ (x1))) or (x1)) and ((x3) and (x2))))) and (x4))) and (x0)) or (x5)) ^ (x2))) ^ ((x4) or ((x5) ^ (x1)))) ^ ((x4) or ((x5) ^ (x1)))) and (x2)) ^ ((x0) and ((x4) or ((x5) ^ (x1))))) ^ (x3))) ^ ((x4) and ((x3) and (x2)))', 'classical_function': 'cf_v6_c20', 'python_function': 'pf_v6_c20'}, '(7,7)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '(not ((((((x6) ^ (x4)) and (x1)) and (x6)) ^ ((x6) ^ (x6))) or ((x2) and ((x0) and (x4))))) and ((x5) ^ (x3))', 'classical_function': 'cf_v7_c7', 'python_function': 'pf_v7_c7'}, '(7,8)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((((((not ((x4) ^ (x0))) and (x2)) ^ (x1)) and (x5)) ^ (((x4) and (x3)) ^ (x6))) or (x2)) ^ (((x4) and (x3)) ^ (x6))', 'classical_function': 'cf_v7_c8', 'python_function': 'pf_v7_c8'}, '(7,9)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '(((((not ((((x0) ^ (x1)) or (x5)) or ((x2) or (x5)))) ^ (x4)) ^ (not (x3))) and (((x2) or (x5)) ^ ((x5) or (x2)))) and (x4)) ^ (x6)', 'classical_function': 'cf_v7_c9', 'python_function': 'pf_v7_c9'}, '(7,10)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((((((not ((((x6) and (x0)) and (x4)) and (x2))) or (x0)) ^ ((x2) and (x6))) or ((x3) ^ (x1))) or (x4)) or (x3)) and ((x5) or ((x4) and (x5)))', 'classical_function': 'cf_v7_c10', 'python_function': 'pf_v7_c10'}, '(7,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': 'not (((((((not ((((x1) ^ (x2)) ^ (x3)) and (x4))) and (x1)) or ((x4) ^ (x5))) or (not (x1))) or ((x3) and (x2))) or (x6)) or ((x0) or (x5)))', 'classical_function': 'cf_v7_c11', 'python_function': 'pf_v7_c11'}, '(7,12)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((not ((((((not ((((x2) ^ (x3)) or (x6)) and (x0))) ^ (((x4) ^ (x0)) or (x4))) or (x3)) or (x1)) or ((x2) or ((x4) ^ (x0)))) ^ (x5))) and (x4)) and ((x4) ^ (x0))', 'classical_function': 'cf_v7_c12', 'python_function': 'pf_v7_c12'}, '(7,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': 'not (not ((not ((((((not ((((x5) or (x1)) and (x5)) ^ (x0))) and (x4)) ^ (x4)) or ((x5) or (x4))) and ((x6) ^ ((x3) and (x5)))) ^ (not (x2)))) ^ ((x6) ^ ((x3) and (x5)))))', 'classical_function': 'cf_v7_c13', 'python_function': 'pf_v7_c13'}, '(7,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': 'not ((((((((((((not ((x1) or (x0))) ^ (x3)) or ((x2) or (x6))) and (x6)) or (x2)) ^ ((x6) or (x3))) ^ ((x4) ^ (not (x1)))) and ((x5) or (x4))) ^ (x2)) and ((x6) or (x3))) and ((x6) or (x3))) or ((x0) ^ (x6)))', 'classical_function': 'cf_v7_c14', 'python_function': 'pf_v7_c14'}, '(7,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((((not (((((((((((x2) or (x1)) ^ (x6)) or (x1)) or (x5)) ^ (x2)) ^ ((x6) or (x2))) and (x3)) ^ (x1)) ^ (x3)) or ((x0) ^ (((x4) or (x6)) ^ (x0))))) ^ (x6)) or ((x4) or (x6))) and (((x4) or (x6)) ^ (not ((x4) or (x6))))) ^ ((x5) or (not ((x4) or (x6))))', 'classical_function': 'cf_v7_c15', 'python_function': 'pf_v7_c15'}, '(7,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '(((((((not (not (((not (((((x1) and (x1)) ^ ((x0) and (x4))) and (x0)) and (x6))) or (x2)) or (x5)))) or (not (x5))) or (x2)) or (not (x5))) ^ ((((x0) and (x4)) ^ (x4)) or ((x5) and ((x0) and (x4))))) and ((x0) and (x4))) ^ (x3)) or (x1)', 'classical_function': 'cf_v7_c16', 'python_function': 'pf_v7_c16'}, '(7,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': 'not (not ((not ((not ((not (not ((((not (((((x0) and (x1)) or ((x4) and (x5))) and (x2)) or (x5))) and (x3)) and ((x6) and (x3))) or (x0)))) and ((x4) and (x5)))) ^ ((x2) or (x4)))) and (not (not ((x4) and (x5))))))', 'classical_function': 'cf_v7_c17', 'python_function': 'pf_v7_c17'}, '(7,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((((not (((((((((not (((not ((x0) ^ (x3))) or (x1)) and ((x6) ^ (x2)))) or (x1)) and (not (x1))) ^ (x1)) ^ (((x6) and (x6)) ^ ((x4) and (not (x1))))) and (((x6) and (x6)) ^ (x5))) or (x3)) and (((x6) and (x6)) ^ (x5))) and ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5))))) or ((((x4) or (x0)) and ((x6) ^ (x2))) or (((x6) and (x6)) ^ (x5)))) and (not (x4))) or ((x1) and ((x6) and (x6)))) ^ (x2)', 'classical_function': 'cf_v7_c18', 'python_function': 'pf_v7_c18'}, '(7,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': 'not ((((((not ((((((not (((((((x3) ^ (x2)) or ((x5) ^ (x3))) and (x4)) or (x6)) ^ ((x2) or (x3))) ^ (x0))) and (x6)) and (x0)) ^ (((x2) or (x3)) and ((x5) ^ (x3)))) and (not (x1))) or (x3))) ^ ((x2) or (x3))) ^ ((x4) and ((((x3) or (x1)) and (x2)) ^ (x5)))) and (not ((x5) ^ (x3)))) and (((x3) or (x1)) and (x2))) or ((x5) and (x3)))', 'classical_function': 'cf_v7_c19', 'python_function': 'pf_v7_c19'}, '(7,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '(((not (((((((not ((((((not ((((x0) ^ (x0)) ^ (x5)) ^ (x0))) ^ (x2)) or (x3)) or (((x6) ^ (x2)) ^ (x6))) and ((x1) or (x5))) or (x3))) ^ (x4)) and (x6)) ^ (x5)) or ((x1) ^ (x3))) ^ (x6)) and ((x1) ^ (x3)))) ^ (not ((x6) ^ (x2)))) ^ (x1)) or ((x5) ^ (x4))', 'classical_function': 'cf_v7_c20', 'python_function': 'pf_v7_c20'}, '(7,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'statement': '((not (not ((((not (not (((((((((((((x3) ^ (x6)) or (x1)) ^ (x5)) ^ (x5)) and ((x0) or (x6))) ^ (x5)) ^ (x1)) or (x1)) or (x4)) or (x2)) and (x6)) or ((x2) and ((x2) ^ (x1)))))) or ((x2) ^ (x1))) ^ ((((x0) or (x6)) or (x5)) and (x4))) and (x6)))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))) or ((((x2) ^ (x1)) ^ (x1)) ^ (x2))', 'classical_function': 'cf_v7_c21', 'python_function': 'pf_v7_c21'}, '(8,8)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(((not (((((x2) or (x0)) or (x2)) and (x2)) and ((x6) and (x7)))) ^ (x4)) ^ ((x4) and ((x3) or (x5)))) or ((x1) or (x3))', 'classical_function': 'cf_v8_c8', 'python_function': 'pf_v8_c8'}, '(8,9)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '((((((not (not ((x7) or (x6)))) and (x5)) ^ (x5)) or (x2)) ^ (x4)) ^ ((x1) ^ ((x6) or (x0)))) ^ (x3)', 'classical_function': 'cf_v8_c9', 'python_function': 'pf_v8_c9'}, '(8,10)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '((not (not (((((((x5) or (x3)) or (x2)) ^ (x5)) and (x1)) or ((x5) ^ (x6))) or (((x6) and (x5)) or (x0))))) or ((x7) or (x6))) ^ (not (((x6) or (x4)) or (x4)))', 'classical_function': 'cf_v8_c10', 'python_function': 'pf_v8_c10'}, '(8,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': 'not (((((((((((x6) ^ (x4)) or (x2)) and (x7)) and (x5)) and ((x5) and (x7))) or ((x3) ^ (x7))) ^ ((x3) ^ (x7))) ^ ((x3) ^ (x7))) and ((x5) and (x7))) ^ ((((x1) and ((x0) or (x1))) and ((x0) or (x1))) or (x6)))', 'classical_function': 'cf_v8_c11', 'python_function': 'pf_v8_c11'}, '(8,12)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(((not ((not (not (((((not (x0)) ^ (not (x3))) or (x5)) and (x4)) and (x0)))) and (x6))) and ((x2) ^ (x1))) or ((x2) ^ (x1))) and ((x1) and (x7))', 'classical_function': 'cf_v8_c12', 'python_function': 'pf_v8_c12'}, '(8,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(not ((((((((((not ((x1) ^ (x7))) and (x3)) ^ (x7)) and ((x5) ^ (x4))) or (x6)) or (x0)) and ((x5) or (x5))) ^ (x2)) or (x4)) and (x5))) or (x1)', 'classical_function': 'cf_v8_c13', 'python_function': 'pf_v8_c13'}, '(8,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': 'not ((((((((((((((x2) ^ (x6)) ^ ((x6) ^ (x2))) and (x3)) or (x2)) and ((x3) ^ (x0))) ^ (x0)) ^ (x3)) and (x1)) and ((x6) ^ (x2))) ^ (x0)) ^ ((x0) ^ (((x3) ^ (x0)) or (x5)))) or (x7)) ^ (x4))', 'classical_function': 'cf_v8_c14', 'python_function': 'pf_v8_c14'}, '(8,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '((((((not ((not (((((not ((x3) and (x6))) and (x3)) ^ (x1)) and (x4)) or (x5))) or (x7))) or ((x3) or ((x4) and (x5)))) ^ ((x5) ^ (not (x5)))) and (x0)) ^ (x6)) or ((x2) and (x5))) or ((not (not (x4))) ^ ((x2) and (x6)))', 'classical_function': 'cf_v8_c15', 'python_function': 'pf_v8_c15'}, '(8,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': 'not (((not (not ((((((not ((((((x4) and (x3)) or (x0)) ^ (x0)) ^ (x5)) and (x4))) or ((x6) or (x3))) ^ ((x6) or (x3))) and (x2)) ^ (x3)) and ((x6) and (x1))))) or ((x1) ^ (x3))) ^ (not (x7)))', 'classical_function': 'cf_v8_c16', 'python_function': 'pf_v8_c16'}, '(8,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '((((not ((((not (((not ((((((x3) or (x3)) ^ (x3)) or (x7)) ^ (x0)) or (x1))) ^ (not (x0))) or ((x4) or ((x4) or (x0))))) ^ ((x2) ^ (x2))) and ((x5) or ((x4) or ((x4) or (x0))))) ^ ((x7) ^ (not (x0))))) or (x4)) or (not (((x5) or (x4)) ^ (x6)))) ^ (x0)) ^ (x7)', 'classical_function': 'cf_v8_c17', 'python_function': 'pf_v8_c17'}, '(8,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(((((not ((((not ((not ((not (((((x7) ^ (x0)) or (x6)) or ((x7) ^ (x3))) ^ (x1))) ^ (x1))) ^ ((x2) ^ (x0)))) and ((x2) ^ (x0))) or (x2)) or (x2))) and (x5)) and (not ((x3) ^ (x1)))) or (not (x4))) ^ (x1)) and ((x1) or (x5))', 'classical_function': 'cf_v8_c18', 'python_function': 'pf_v8_c18'}, '(8,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(((not ((not (((((not ((((not (((((x0) or (x2)) ^ (x0)) or (x4)) or (x4))) or (x5)) ^ ((x3) or ((x3) or ((x3) or (x6))))) or ((x1) or (x3)))) ^ (x1)) ^ (((x1) or (x3)) ^ (x6))) and (x7)) or (x7))) or ((x3) or ((x3) or (x6))))) and ((x3) or ((x3) or ((x3) or (x6))))) and ((x6) and (x4))) and (x1)', 'classical_function': 'cf_v8_c19', 'python_function': 'pf_v8_c19'}, '(8,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '((((((((((not (((not ((((not (((x2) and (x2)) and (x5))) or (x6)) and (x7)) and (x5))) and (x2)) ^ ((x1) and (x3)))) and (x3)) or (x5)) and (x7)) and ((x1) and (x4))) and (x3)) and ((x3) and ((x0) or (not (x2))))) ^ (x4)) and (((x1) and (x3)) or (((x0) or (not (x2))) and (x2)))) and (x7)) ^ ((x0) or (not (x2)))', 'classical_function': 'cf_v8_c20', 'python_function': 'pf_v8_c20'}, '(8,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(not ((((not ((((((not ((not ((not (((not (((x6) ^ (x3)) ^ (x4))) ^ (x2)) and (x2))) and ((x1) ^ (x7)))) and (x4))) and (not (x0))) or (x0)) and ((x5) ^ (x7))) ^ ((not (x7)) and (x2))) ^ (x2))) and (not (x1))) ^ (((x5) ^ (x7)) and (((x3) and (x2)) or (x6)))) ^ ((x3) and (x2)))) and ((not (x5)) and (x7))', 'classical_function': 'cf_v8_c21', 'python_function': 'pf_v8_c21'}, '(8,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], 'statement': '(((not ((((not ((((((((not (not ((((not ((x7) or (x0))) ^ (x4)) and ((x4) or (x0))) or (x5)))) ^ ((x5) or (x1))) or (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) or (x5)) ^ (x2)) and (((x4) or (x0)) ^ (x6))) or (x6)) and (x1))) and (((x4) or (x0)) ^ (((x4) or (x0)) ^ (x6)))) ^ (not (x7))) ^ (((x1) and (x7)) ^ (x4)))) or ((not (x0)) ^ (not ((x4) or (x3))))) and (((not (((x4) or (x0)) ^ (x6))) or (x0)) ^ ((x5) or (x1)))) ^ ((x1) or (x2))', 'classical_function': 'cf_v8_c22', 'python_function': 'pf_v8_c22'}, '(9,10)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((((((((((x5) or (x4)) and (x3)) and (x4)) or (x7)) or ((x8) and (x8))) or (x8)) ^ ((x8) or (x5))) and (x2)) ^ ((x8) and (x8))) and ((x6) ^ ((x0) and (x1)))', 'classical_function': 'cf_v9_c10', 'python_function': 'pf_v9_c10'}, '(9,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(not ((((((((not ((x2) or (x1))) and (not ((x6) or (x5)))) ^ (not ((x6) or (x5)))) and (x3)) and (x0)) ^ (x5)) ^ (x4)) or (((x0) and (x7)) and ((x4) ^ (x8))))) and (((x0) and (x7)) ^ (x8))', 'classical_function': 'cf_v9_c11', 'python_function': 'pf_v9_c11'}, '(9,12)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((not ((((((((((x8) or (x8)) and (x6)) and (x0)) and (x8)) ^ (x4)) and ((x8) and (x8))) or (x4)) ^ (x3)) or (x7))) or ((x5) ^ (((x1) or (x6)) and (x1)))) and (x2)', 'classical_function': 'cf_v9_c12', 'python_function': 'pf_v9_c12'}, '(9,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(not ((not (((not ((((not (((x0) or (x2)) or (x3))) and (x5)) and (x0)) or (x6))) and ((x7) and (x3))) and (((x8) or (x8)) and ((x3) or (x3))))) or ((x6) and (x1)))) and ((x8) or ((x4) and (x2)))', 'classical_function': 'cf_v9_c13', 'python_function': 'pf_v9_c13'}, '(9,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(((((((((((((not (x4)) and (x5)) ^ (x8)) or (x3)) and ((x7) or (x7))) and (x5)) and (x8)) or (x6)) or (x4)) or (x4)) or ((x7) or (x7))) or ((x7) or (x7))) ^ (((x2) ^ (x1)) or (x0))) or ((x6) and (((x7) or (x7)) and (x2)))', 'classical_function': 'cf_v9_c14', 'python_function': 'pf_v9_c14'}, '(9,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(((not ((((((not ((not ((((x6) ^ (x5)) or (x2)) and (x6))) or (not (x3)))) ^ (x2)) and ((x2) or (x4))) or ((x0) and (x4))) or ((x1) or ((x0) or (x2)))) ^ (((x8) and ((x0) and (x4))) or (not (x3))))) ^ (x7)) ^ (x2)) and (x8)', 'classical_function': 'cf_v9_c15', 'python_function': 'pf_v9_c15'}, '(9,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((((((((((not (not ((not (((x5) ^ (x4)) and (x7))) or ((x6) ^ (x6))))) and (x2)) or (not (x3))) or (((x5) and (x8)) or (not (x3)))) or ((x0) or (not (x8)))) or (x2)) and (not ((x5) and (x8)))) or ((x0) or (not (x8)))) ^ ((not (x8)) or (not (x8)))) and (x2)) and ((x1) or (not (x3)))', 'classical_function': 'cf_v9_c16', 'python_function': 'pf_v9_c16'}, '(9,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(not (not (((((((not ((((((not ((x2) and (x0))) or (x2)) and (x6)) ^ ((x5) and (x5))) and ((x4) or (x8))) and (not (x3)))) or (x7)) and (not (x3))) or (x1)) ^ (x2)) or ((x5) and (x5))) or ((x4) or (x8))))) or ((x1) or (x2))', 'classical_function': 'cf_v9_c17', 'python_function': 'pf_v9_c17'}, '(9,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((((not ((((((not ((not (not (not ((((x6) or (x5)) or (x4)) or (x7))))) and (x8))) and (x3)) or (x6)) or (not (x2))) and (not (x1))) or (not (x5)))) and ((x6) or (((x1) and (x4)) and (x0)))) ^ (x6)) and ((x4) or ((x4) or (x7)))) ^ (not (x2))', 'classical_function': 'cf_v9_c18', 'python_function': 'pf_v9_c18'}, '(9,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(((((((((((((not (not (((((x8) or (x1)) or (x1)) or (x2)) and (x7)))) and (not (x3))) and (not (x0))) ^ (x5)) and (x4)) ^ (x5)) ^ ((x7) or (x2))) and (x7)) and (((x4) and (x7)) and ((x4) and (x7)))) or (((x1) or (x6)) and ((x1) or (x6)))) and ((((x1) or (x6)) and (x2)) or (((x1) or (x6)) and ((x1) or (x6))))) ^ (x3)) ^ (x6)) ^ (not (x3))', 'classical_function': 'cf_v9_c19', 'python_function': 'pf_v9_c19'}, '(9,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((not (((not (((((not ((((((((((x0) and (x6)) or ((x0) and (x6))) or (x5)) or ((x0) and (x6))) ^ (x3)) ^ (x0)) or (x2)) or ((x5) or (x8))) and ((((x2) or (x4)) ^ (x4)) or (x4)))) and ((((x2) or (x4)) ^ (x4)) or (x4))) or (x7)) ^ (x2)) ^ (((x2) or (x4)) or ((x0) and (x6))))) and (x8)) or ((((x2) or (x4)) ^ (x4)) or (x4)))) ^ (x3)) and ((x4) and (((((x2) or (x4)) ^ (x4)) or (x4)) and (x1)))', 'classical_function': 'cf_v9_c20', 'python_function': 'pf_v9_c20'}, '(9,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((((((not ((not (((((not (((((not (((x1) and (x6)) ^ (x5))) and ((x3) and (x1))) and (x3)) ^ (x0)) and (x7))) ^ (x4)) or (x1)) or (x7)) ^ (not (x3)))) ^ ((x2) and (x5)))) and (x6)) ^ (x0)) or ((((x3) and (x1)) ^ ((x3) and (x1))) and ((x2) and (x5)))) or (x2)) ^ (x3)) or ((x8) and ((x3) and (x1)))', 'classical_function': 'cf_v9_c21', 'python_function': 'pf_v9_c21'}, '(9,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '((((((((((not (not ((not (((not ((((((x2) or (x7)) and (x7)) or (x2)) or (x3)) and ((x4) or ((x6) and (x1))))) or (not (x6))) or (x0))) and (x7)))) or (not (not (x6)))) or ((x6) and (x1))) ^ (x2)) or (x5)) or (not (x4))) and (not (not (x6)))) and (x4)) and (((x3) and (x1)) ^ ((x7) or (x0)))) and ((x5) and (x7))) and ((((x8) and (x3)) or (x6)) or ((((x8) and (x3)) or (x6)) or ((x6) and (x1))))', 'classical_function': 'cf_v9_c22', 'python_function': 'pf_v9_c22'}, '(9,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], 'statement': '(not (((((((not (not ((not (not ((((((((not (((x6) or (x1)) ^ (x7))) or (x4)) and (x8)) and ((x0) and (x7))) and (x5)) and ((x0) and (x7))) ^ (x3)) ^ ((x6) and ((x2) or (x0)))))) ^ ((x2) ^ (x5))))) or (x1)) or (((x2) ^ (x5)) and (x1))) and (((x2) or (x0)) or (x0))) ^ ((x1) ^ (x4))) or (not (not (x8)))) and (not (x1)))) ^ (((x2) or (x0)) or (x0))', 'classical_function': 'cf_v9_c23', 'python_function': 'pf_v9_c23'}, '(10,13)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((((not (((((((((x0) ^ (x4)) ^ (x5)) and ((x8) ^ (x2))) and ((x3) ^ ((x8) ^ (x2)))) and (x3)) and ((x3) ^ ((x8) ^ (x2)))) or (x7)) ^ (x4))) or ((x3) ^ ((x8) ^ (x2)))) ^ (not (x1))) ^ ((x9) and (x6))) ^ (x1)', 'classical_function': 'cf_v10_c13', 'python_function': 'pf_v10_c13'}, '(10,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((((not (((((not ((((((x1) ^ (x0)) ^ (x8)) ^ (x4)) ^ (x4)) ^ (x1))) ^ (x2)) ^ (x3)) and (x9)) or (x7))) and (x3)) or (((x4) ^ (x6)) or (x9))) ^ ((x4) ^ (x9))) ^ (((x5) and ((x4) ^ (x6))) and (x5))', 'classical_function': 'cf_v10_c15', 'python_function': 'pf_v10_c15'}, '(10,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((not (((((((((not ((((not (x6)) or (not (x4))) ^ (x6)) ^ (x1))) or (x0)) and (x1)) or (not ((x3) or (x9)))) and (x2)) or (x7)) or (x5)) ^ (x5)) or (not ((x3) or (x9))))) or (x8)) ^ ((x2) or (not ((x3) or (x9))))', 'classical_function': 'cf_v10_c16', 'python_function': 'pf_v10_c16'}, '(10,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((not (((((not ((((((((((x4) and (x9)) ^ (x1)) or (x9)) or ((x3) or (x2))) or (x5)) ^ (x7)) and (x2)) and ((x5) or (x5))) or ((x8) and (x5)))) and (x7)) and (x9)) or ((x6) ^ ((x3) and (x2)))) ^ (x1))) and (x5)) or (((x0) and (x8)) or (((x6) ^ ((x3) and (x2))) ^ ((x1) or (x5))))', 'classical_function': 'cf_v10_c17', 'python_function': 'pf_v10_c17'}, '(10,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': 'not (((((((((not (not ((((((((x2) or (x5)) ^ (x9)) or (x8)) ^ (x6)) ^ (x4)) or ((x6) ^ ((x2) ^ (x6)))) or (x7)))) and (x1)) and ((x6) ^ (x3))) ^ (((x4) or (x6)) and (not ((x6) ^ ((x2) ^ (x6)))))) or (x0)) ^ (x3)) or (x1)) ^ (x3)) and ((x2) or (x0)))', 'classical_function': 'cf_v10_c18', 'python_function': 'pf_v10_c18'}, '(10,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': 'not (not ((not ((((((((((((((((x9) or (x5)) and (x4)) ^ (x5)) or (x4)) and (x1)) and (x7)) and (x2)) ^ ((x9) ^ (x1))) or (x0)) and ((x8) or (x3))) or (x4)) ^ ((x1) and (x2))) ^ ((x9) ^ (x1))) ^ ((x4) and ((x8) or (x6)))) and (((x8) or (x3)) and (x7)))) and (x5)))', 'classical_function': 'cf_v10_c19', 'python_function': 'pf_v10_c19'}, '(10,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '(not ((not (not (not (not (((((not (((not (not ((not (((x2) and (x5)) and (x6))) ^ (x4)))) ^ (x8)) ^ (x2))) ^ ((x0) and (x1))) or ((x9) ^ (x3))) or (x8)) and (not (x4))))))) ^ ((x6) and (((x0) and (x1)) ^ (x7))))) or ((x9) ^ (x3))', 'classical_function': 'cf_v10_c20', 'python_function': 'pf_v10_c20'}, '(10,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '(((((not ((((not ((((((((((((x8) and (x5)) and (x6)) and (x2)) or (x1)) and ((x6) and (x3))) and (x5)) and (x9)) and (((x6) and (x3)) and (x5))) or (((x9) and (x8)) and (x1))) and ((x6) and (x3))) and (x4))) ^ ((x6) or (x7))) ^ (((x9) and (x8)) and (x1))) and (x1))) and ((x1) or (x4))) ^ (x3)) ^ (x7)) ^ ((x9) ^ (x0))) and (x3)', 'classical_function': 'cf_v10_c21', 'python_function': 'pf_v10_c21'}, '(10,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '(((((((not ((not (((((((((((((x1) ^ (x3)) and (x2)) ^ (x2)) ^ (x1)) and (x4)) or ((x9) and ((x7) ^ (x9)))) and (x2)) ^ ((x3) ^ (x6))) or (x2)) or (x9)) or ((x9) and ((x7) ^ (x9)))) and (((x0) or (x4)) or (x9)))) ^ ((x7) ^ (x9)))) or (x6)) or (x8)) ^ (x4)) and (((x7) ^ (x9)) ^ (x5))) ^ (((x3) and (x4)) or (x9))) or (((x0) or (x4)) or (x9))) and ((x9) ^ (x4))', 'classical_function': 'cf_v10_c22', 'python_function': 'pf_v10_c22'}, '(10,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((((((not (((((((not (not (not (((((not (((x0) and (x2)) ^ ((x6) and (x0)))) or (x5)) and ((x8) ^ (x1))) ^ (x8)) ^ (x4))))) ^ ((x9) or (x9))) or (not (x0))) ^ ((x7) ^ (x6))) and (((x6) ^ (x8)) ^ (x9))) and (((x8) ^ (x1)) and (x9))) and (x9))) or (x3)) ^ (x9)) ^ (((x9) and (x4)) ^ (x5))) and (x7)) or (not (x4))) and ((x6) ^ (x8))', 'classical_function': 'cf_v10_c23', 'python_function': 'pf_v10_c23'}, '(10,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'statement': '((((not (not (((not (((not (((((((((((not ((x5) ^ (x5))) and (not (x1))) and (x8)) and ((x6) ^ (x7))) or ((x5) and (x9))) or (x2)) or (x3)) and (x5)) and ((x6) ^ (x7))) and ((x2) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) or (x9))) or (x9)) or (((x6) ^ (x7)) and ((x6) ^ (x7))))) ^ (((x6) ^ (x7)) and ((x6) ^ (x7)))) ^ (x1)))) or ((x8) ^ ((x1) or (x0)))) or (not ((not (x1)) ^ (x5)))) ^ (x4)) or (x6)', 'classical_function': 'cf_v10_c24', 'python_function': 'pf_v10_c24'}, '(11,11)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(not (((((not (((((x0) and (x0)) and (x10)) and (x6)) and (x2))) and (x4)) ^ (x5)) ^ (((x6) and (not ((x2) and (x10)))) ^ (not ((x2) and (x10))))) and (((x3) and (x7)) and (x9)))) and ((x8) ^ (x10))', 'classical_function': 'cf_v11_c11', 'python_function': 'pf_v11_c11'}, '(11,14)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(not ((((((((not (((((x4) and (x9)) ^ (x3)) ^ (x8)) ^ ((x5) ^ (x3)))) or (not (x10))) or (x3)) or (x6)) ^ (x2)) ^ (x8)) or ((((x8) ^ (x6)) or (x0)) or (x7))) or (x4))) or (x1)', 'classical_function': 'cf_v11_c14', 'python_function': 'pf_v11_c14'}, '(11,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(((not ((((not (not ((((not (((x5) and (x0)) ^ (x2))) or (x3)) and (x4)) or (x6)))) ^ (((x3) ^ (x8)) or ((x7) and (x6)))) or (((x3) ^ (x8)) or ((x7) and (x6)))) and (((x3) ^ (x8)) or ((x7) and (x6))))) ^ (x3)) and (x9)) or (x10)', 'classical_function': 'cf_v11_c15', 'python_function': 'pf_v11_c15'}, '(11,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(not (((((not ((not (((((((not (x6)) ^ (x10)) and (x3)) and (x4)) and (x3)) or (x8)) and (x8))) and (x2))) ^ ((x10) and (x5))) ^ (not (x9))) and ((x10) ^ (x3))) or (x7))) or (x0)', 'classical_function': 'cf_v11_c16', 'python_function': 'pf_v11_c16'}, '(11,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(((((not (not (((((not ((not ((((x6) or (x3)) and (x3)) or (x7))) ^ (x9))) and (x5)) or ((x4) ^ (x3))) ^ ((x2) or (x10))) ^ (((x10) and ((x9) and (x2))) ^ (x1))))) or (not ((x6) or (x4)))) or (x3)) ^ (x0)) ^ (((x10) and ((x9) and (x2))) ^ (x1))) or ((x8) or (x5))', 'classical_function': 'cf_v11_c17', 'python_function': 'pf_v11_c17'}, '(11,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '((((not (not (((((not (((not ((not (((x5) and (x5)) ^ (x8))) ^ ((x0) and (x6)))) and (x7)) ^ (x3))) or ((x0) and (x6))) or ((x4) ^ (x4))) ^ (x7)) ^ (not (x4))))) or ((not (x10)) or (x9))) ^ ((x0) and (x6))) or ((x7) and ((x3) ^ (x2)))) ^ ((x3) or (x9))', 'classical_function': 'cf_v11_c18', 'python_function': 'pf_v11_c18'}, '(11,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '((((((((not (not ((not ((((not ((((x5) or (x5)) ^ ((x0) or (x7))) or (x6))) or (x0)) or (x7)) ^ (x10))) ^ (x4)))) ^ ((x2) ^ (x10))) or (x8)) or (x8)) or (not (x0))) ^ (((not (x9)) ^ (x2)) or (not (x10)))) or ((((x0) or (x7)) or (x9)) or (x3))) ^ (not (x9))) and (x6)', 'classical_function': 'cf_v11_c19', 'python_function': 'pf_v11_c19'}, '(11,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(((not (((((not ((((((((((((x0) and (x3)) and (x5)) and (x4)) ^ (x0)) ^ (x3)) ^ (x5)) and ((x9) and (x10))) or (x10)) and ((x3) and (x6))) or (x10)) and ((x8) ^ (x1)))) and (((x9) and (x10)) and ((x3) ^ (x8)))) and (((x8) ^ (x1)) or (x3))) or (x2)) ^ (x7))) or (((x8) ^ (x1)) or (x3))) ^ ((x4) and ((x3) and (x6)))) and (x0)', 'classical_function': 'cf_v11_c20', 'python_function': 'pf_v11_c20'}, '(11,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': 'not (((((not (not (((((((((not ((not (((not (x9)) or (x2)) or (x5))) or (x6))) or (x0)) ^ ((x10) or (x8))) or ((x1) ^ (x3))) or (x1)) and (x3)) ^ (x7)) or ((x7) and (x1))) ^ ((x7) and (x1))))) ^ ((((x1) ^ (x3)) or (x1)) or (x4))) and (not (x8))) and ((x7) and (x1))) and ((x10) ^ (x3)))', 'classical_function': 'cf_v11_c21', 'python_function': 'pf_v11_c21'}, '(11,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(((not ((not (not (((not (((not ((not ((not ((((((x10) or (x9)) or (x2)) ^ (x6)) or (x1)) ^ (x9))) or (x1))) and (not (x8)))) or ((x9) or ((x5) or ((x8) or (x0))))) ^ (x4))) or (x3)) or (x1)))) or (not ((x5) or ((x8) or (x0)))))) or (x0)) and ((x7) or (x8))) ^ (x9)', 'classical_function': 'cf_v11_c22', 'python_function': 'pf_v11_c22'}, '(11,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': 'not (not ((((((not ((not (((((((((((not (((x6) or (x5)) or (x7))) ^ (x7)) or (x8)) and (x3)) and (x7)) and ((x1) or (x6))) ^ (x3)) ^ (x7)) and (x3)) or (x2)) ^ ((((x5) or (x5)) and (not (x0))) and (x9)))) ^ (not ((x9) ^ ((x10) and (x6)))))) or (((x5) or (x5)) and (not (x0)))) or (x4)) ^ ((x2) ^ (x5))) and (x8)) ^ (((x0) or (x0)) or (x5))))', 'classical_function': 'cf_v11_c23', 'python_function': 'pf_v11_c23'}, '(11,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '(((not (not (((((((((((not ((not (((((not ((x10) and (x7))) ^ ((x3) and (x0))) or (x0)) or (not (x9))) ^ ((x2) or (x2)))) and ((x6) or (x7)))) or (x7)) or ((x3) and (x0))) ^ ((x10) or (x1))) ^ (x5)) or (not (x7))) and (not (x9))) ^ (not (x8))) ^ (((x2) or (x2)) and (x8))) or ((not (x8)) ^ (x10))) and (((x2) or (x2)) ^ (x6))))) and (x8)) and (x4)) or ((x3) ^ (not (x9)))', 'classical_function': 'cf_v11_c24', 'python_function': 'pf_v11_c24'}, '(11,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], 'statement': '((((((((((((not (((((((((not ((((x7) and (x6)) or ((x2) and (x3))) or (x1))) ^ (x5)) and (x4)) or ((x5) or (x6))) and ((x5) or (x6))) and ((x2) and (x3))) ^ (x2)) or ((x2) and (x3))) ^ (x9))) and ((x8) ^ (x0))) or (x10)) ^ (not (x8))) and ((x5) or (x6))) and ((x8) ^ (x0))) ^ (((x2) and (x3)) ^ (x9))) ^ ((x4) and (x10))) or ((((x8) or (x2)) and (x8)) ^ ((x5) or (x6)))) ^ (((x8) or (x2)) and (x8))) ^ (((x8) ^ (x0)) ^ (x9))) ^ (((x4) or (x1)) ^ (x6))) ^ (((x8) ^ (x0)) ^ (x9))', 'classical_function': 'cf_v11_c25', 'python_function': 'pf_v11_c25'}, '(12,15)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((((((((((((((not (x10)) ^ (x7)) or (x0)) or ((x5) or (x1))) and (x9)) or (x9)) ^ (x8)) ^ (((x5) or (x1)) and (not (x3)))) and (x2)) or (x4)) and (x6)) ^ (x3)) or (x8)) ^ (x9)) or (x11)', 'classical_function': 'cf_v12_c15', 'python_function': 'pf_v12_c15'}, '(12,16)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((((((((((not ((((((x4) or (x4)) and (x1)) and (x4)) or (x6)) and (x1))) ^ ((x8) and ((x0) or (x11)))) ^ (x6)) ^ (x3)) and (not (x2))) ^ (x7)) ^ ((x4) and (x9))) and (x8)) or ((x10) or ((x8) and ((x0) or (x11))))) or ((((((x10) or ((x8) and ((x0) or (x11)))) and (x8)) ^ (x1)) and (x3)) and (x5))) and (x3)', 'classical_function': 'cf_v12_c16', 'python_function': 'pf_v12_c16'}, '(12,17)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((not (((((((((((((not ((x7) ^ (x5))) and (x3)) or (x1)) and (x1)) or (x11)) and (x5)) and (x0)) or ((x2) and (x3))) or (((x5) ^ (x10)) and (x4))) and (x7)) and (x2)) ^ ((x6) or (x5))) or ((((x5) ^ (x10)) or (not (x10))) and (x9)))) ^ (x7)) or (x8)', 'classical_function': 'cf_v12_c17', 'python_function': 'pf_v12_c17'}, '(12,18)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '(((((((not ((not ((((not (((((x6) ^ (x4)) or (x11)) or (x3)) or (x7))) and (x1)) or ((x0) or ((x11) or (x11)))) ^ ((x5) or (x7)))) or ((x10) ^ (x6)))) ^ (x7)) and ((x4) or (x6))) or ((x0) or ((x11) or (x11)))) or (((x9) and (x2)) ^ ((x6) or (x11)))) or (x8)) or (((x9) and (x2)) ^ ((x6) or (x11)))) ^ (x10)', 'classical_function': 'cf_v12_c18', 'python_function': 'pf_v12_c18'}, '(12,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((not (((((((((((((((((x1) or (x11)) ^ (x1)) or (x0)) or (x6)) and (x8)) ^ ((x5) or (x4))) ^ (x1)) ^ ((x1) or (x2))) and (x7)) or (x1)) and (x7)) ^ (x10)) or ((x1) and ((x4) ^ (x6)))) or ((x6) or (((x10) ^ ((x5) or (x4))) or (x5)))) or ((x1) and ((x4) ^ (x6)))) and (x3))) and (x6)) ^ (x9)', 'classical_function': 'cf_v12_c19', 'python_function': 'pf_v12_c19'}, '(12,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((not (((not ((((((((not (((((((x0) ^ (x9)) ^ (x7)) ^ (x4)) and (x3)) or (x0)) and (x0))) and (x6)) ^ (x2)) or ((x11) ^ (x3))) or (x7)) or (x0)) and (x5)) ^ ((x2) ^ (x8)))) or ((x0) ^ (not (x8)))) ^ (x11))) ^ (((x7) and (x1)) or (x0))) or ((x8) ^ (x10))', 'classical_function': 'cf_v12_c20', 'python_function': 'pf_v12_c20'}, '(12,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '(not (not (((((not ((((((((not (((((not (x11)) or (x4)) or (x9)) and ((x0) or (x3))) and (x2))) or (x5)) ^ ((x7) and (x11))) or (x1)) and (x5)) or (x4)) and ((x0) or (x3))) and ((x7) and (x10)))) or ((x0) and (x8))) ^ ((x0) and (x8))) ^ ((x2) or (x6))) ^ (not ((x2) or (x6)))))) ^ (x0)', 'classical_function': 'cf_v12_c21', 'python_function': 'pf_v12_c21'}, '(12,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((((not (not (not (((not ((((not (((((((not ((x1) and (x10))) or (x5)) and (x3)) and ((x4) or (x1))) and (x9)) ^ (x10)) ^ ((x11) and (not (x9))))) or ((x2) ^ (x2))) or (x6)) or ((x2) ^ (x2)))) or (x1)) ^ (not (x9)))))) or ((x1) ^ ((x4) or (x1)))) and (x8)) ^ ((x7) or (x0))) and ((x6) and ((x11) and (not (x9))))', 'classical_function': 'cf_v12_c22', 'python_function': 'pf_v12_c22'}, '(12,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '(not (not (not ((((not ((((not ((((((((((((x3) ^ (x5)) and (x9)) ^ (x0)) and (x6)) ^ (x7)) and (x6)) ^ (x3)) or (x8)) or (x3)) or (((x2) ^ (x9)) and (x10))) and (x9))) or (((x2) ^ (x9)) ^ (x4))) or (x7)) or (x5))) ^ ((((x2) ^ (x11)) or (x4)) and (x4))) or (x2)) or ((x2) ^ (x11)))))) ^ (x11)', 'classical_function': 'cf_v12_c23', 'python_function': 'pf_v12_c23'}, '(12,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': 'not (not ((((((((((((((((not (not ((((((x10) or (x8)) or (x5)) and (x6)) or (x0)) ^ (x4)))) or (not (x9))) ^ ((x7) ^ (x3))) or (x4)) or (x9)) and (not ((x2) and ((x5) or (x3))))) and (x0)) and (x7)) or (x0)) or ((x5) or (x7))) ^ (((x5) or (x3)) or (x10))) or ((x5) or (x3))) ^ ((x11) and (x5))) ^ (not (x9))) or ((x5) or (x7))) and ((x9) ^ (x9))))', 'classical_function': 'cf_v12_c24', 'python_function': 'pf_v12_c24'}, '(12,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((((not (((((((not ((((not ((not (((((((not (x10)) and (x6)) and (x9)) ^ (x8)) and (x1)) and (x0)) and ((x0) and (x2)))) and (not (x1)))) ^ ((x6) and (x11))) ^ (x4)) or (x1))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (((x6) and (x11)) or ((x5) and (x6)))) ^ (x7)) or (x3)) and ((not (x1)) ^ ((x8) and (x11)))) ^ ((not (x6)) ^ (x1)))) or ((x0) and (x2))) or ((x6) and (x11))) ^ ((x6) and (x11))) and ((x5) and (x6))', 'classical_function': 'cf_v12_c25', 'python_function': 'pf_v12_c25'}, '(12,26)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'statement': '((((((((not (not ((not ((not ((((((not ((not ((((not (x10)) or (x5)) and (x5)) or ((x0) and (x4)))) ^ (x1))) ^ (x1)) and ((x5) or (x3))) or ((x3) or (x11))) and (x11)) ^ ((x3) or (x11)))) ^ (x8))) and ((x4) ^ (x11))))) and (x2)) or ((x7) ^ (x8))) or (x3)) or (x6)) ^ (((x5) ^ (not (x0))) or (((x5) ^ (not (x0))) or ((x5) or (x3))))) and (x8)) ^ (x10)) or (x9)', 'classical_function': 'cf_v12_c26', 'python_function': 'pf_v12_c26'}, '(13,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': '(((((((((not (((((not (((((((x4) ^ (x10)) and ((x4) ^ (x2))) or (x6)) ^ ((x4) ^ (x2))) ^ (x0)) ^ (x8))) ^ (x11)) and (x6)) and ((x5) ^ ((x10) or (x5)))) and (x11))) ^ (x7)) or (not (x2))) and (x0)) or (((x1) ^ (x11)) and ((x5) ^ ((x10) or (x5))))) or (x9)) ^ (x4)) ^ ((x5) and (x3))) or (x12)) ^ (x9)', 'classical_function': 'cf_v13_c21', 'python_function': 'pf_v13_c21'}, '(13,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': '(((((not (((not (((not (((((not ((((((x4) and (x5)) ^ (x7)) ^ (x10)) and (x3)) or (x6))) ^ (x2)) and (x4)) or ((x4) ^ (x0))) ^ (x11))) or (x8)) ^ (x12))) ^ (x7)) or (((x10) ^ (x2)) ^ ((x3) or (x7))))) and ((x10) ^ (x2))) and ((x4) or (x11))) and ((x1) and (x6))) and (x9)) and (x7)', 'classical_function': 'cf_v13_c22', 'python_function': 'pf_v13_c22'}, '(13,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': '((((not ((not ((((((not (not (not (not (not ((not (((((x0) and (x4)) or (x5)) and ((x10) or (x11))) and (x6))) or (x11))))))) and (x10)) or ((x6) and ((x6) and (x9)))) or (x12)) or (not ((x8) and (x11)))) and (not ((x10) or (x11))))) or ((x6) and ((x6) and (x9))))) ^ ((not (x7)) or (x12))) ^ ((x3) or (x12))) and (not (x8))) ^ ((x2) or (x8))', 'classical_function': 'cf_v13_c23', 'python_function': 'pf_v13_c23'}, '(13,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': 'not (((((not ((((((((not ((((not (not (not (((not ((x8) and (x7))) ^ ((x9) and (x3))) and (x4))))) or (x10)) ^ (x7)) and (x2))) ^ (not (x5))) ^ (not (x8))) and (x0)) ^ (x12)) or (not (x11))) ^ (not (x6))) or (((x10) or (x8)) and (x7)))) ^ (((x10) or (x8)) and (x7))) ^ (x2)) ^ (x5)) or (x3))', 'classical_function': 'cf_v13_c24', 'python_function': 'pf_v13_c24'}, '(13,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': '((not (((((((((not (not ((((((not ((((((not (x5)) or (x3)) and (x6)) or (x4)) ^ (x12)) and (x0))) or (x2)) and (x7)) or (((x6) and ((x4) or (x11))) or (x4))) and ((x6) or (x1))) ^ (x5)))) ^ (((x6) and ((x4) or (x11))) and ((x6) and ((x4) or (x11))))) and ((x6) or (x1))) and (((x4) or (x11)) ^ (x8))) or ((x0) and ((x11) ^ (x12)))) and (x3)) or (not (x9))) and ((x0) and ((x11) ^ (x12)))) ^ (x5))) and ((x5) and (not (x12)))) and ((x9) and ((((x4) or (x11)) ^ (x8)) or (((x4) or (x11)) and (x10))))', 'classical_function': 'cf_v13_c25', 'python_function': 'pf_v13_c25'}, '(13,26)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': 'not (((not (((((((((((not ((not (((not (((((((x8) ^ (x7)) and (x1)) ^ (x11)) or (x6)) ^ (x3)) or (x6))) ^ ((x6) ^ (x1))) and ((x11) or (x11)))) and ((x6) ^ (x1)))) ^ (x7)) and ((x11) or (x11))) ^ (x8)) ^ ((x2) and (x10))) or (x3)) ^ (x12)) or (((x6) ^ (x1)) ^ (x9))) and (x12)) or (((x4) and (x9)) ^ (x11))) or (not (x11)))) ^ (((x6) or ((x5) ^ (x5))) and (not (x6)))) ^ (((x6) ^ (x1)) and (x0)))', 'classical_function': 'cf_v13_c26', 'python_function': 'pf_v13_c26'}, '(13,27)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'], 'statement': 'not (((((((((not ((((((not (((((not ((((not (not ((x0) ^ (x4)))) ^ (x3)) or (not (x8))) or (x11))) ^ ((x9) or (x7))) ^ (x3)) ^ ((x9) or (x7))) ^ (x12))) or (not (x2))) and (x5)) and (x6)) and ((x1) ^ (x9))) or (x8))) ^ (x2)) and ((x3) or ((x10) ^ (x4)))) and (x5)) or (x10)) or (x10)) or ((x0) or (not (not (x12))))) or (((x3) or ((x10) ^ (x4))) and (x0))) and ((x0) or (not (not (x12)))))', 'classical_function': 'cf_v13_c27', 'python_function': 'pf_v13_c27'}, '(14,20)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '(((not (((((((((((((((((x7) and (x11)) ^ (x1)) ^ (x6)) ^ (x9)) and (x11)) ^ (x13)) or (x11)) and (x12)) ^ (x10)) ^ (x1)) or ((x2) or (x1))) and (x3)) or (((x9) and (x5)) ^ ((x4) ^ (x0)))) ^ ((x7) ^ (x8))) or ((x4) ^ (x6))) or (x8))) or (x4)) ^ (((x9) and (x9)) or ((x7) ^ (x8)))) ^ ((x7) and (x4))', 'classical_function': 'cf_v14_c20', 'python_function': 'pf_v14_c20'}, '(14,21)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '((((not (((((((((((((((not ((x3) and (x5))) ^ (x0)) and (x8)) or (x8)) ^ ((x10) and (x12))) or (x13)) ^ (x6)) or (x9)) or ((x11) or ((x6) ^ (x2)))) and (x4)) ^ (x0)) ^ ((x1) and (x5))) and (x10)) or ((x7) or (x10))) or ((x11) ^ ((x5) and (x5))))) or ((x1) or ((x2) ^ (x5)))) ^ (x10)) ^ (x8)) and ((x1) ^ ((x2) or ((x2) ^ (x5))))', 'classical_function': 'cf_v14_c21', 'python_function': 'pf_v14_c21'}, '(14,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '(((((((((((((not (((((not ((((x10) ^ (x12)) ^ (x8)) and (x0))) or (x4)) ^ (x8)) ^ (x5)) or (x1))) ^ (x6)) and (x13)) and ((x3) and (x5))) ^ (x1)) and (x9)) and (x5)) ^ (((x3) and (x5)) ^ ((x13) ^ (x2)))) or ((x10) ^ (x3))) or (((x12) ^ (x11)) and (x3))) and ((x0) or ((x13) ^ (x2)))) ^ ((x13) ^ (x2))) ^ (((x10) ^ (x7)) and ((((x3) and (x5)) ^ ((x13) ^ (x2))) or (x5)))) and ((x6) ^ (not (x6)))', 'classical_function': 'cf_v14_c22', 'python_function': 'pf_v14_c22'}, '(14,23)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '(not ((((not (((((((((not (not ((not (((((not (x8)) or (x11)) ^ (x2)) or (x1)) ^ (x12))) ^ (x2)))) ^ (x12)) and (x1)) or (((x1) ^ (x6)) and (x7))) ^ (x0)) and (x5)) and ((x1) ^ (x6))) ^ (x10)) or ((x11) ^ (x13)))) and (x11)) ^ (x3)) ^ (((x10) or (x11)) ^ ((not (x9)) ^ (x4))))) ^ (x11)', 'classical_function': 'cf_v14_c23', 'python_function': 'pf_v14_c23'}, '(14,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '((((not (not (not ((((not ((((not (not (((((not ((((x6) and (x13)) or (x10)) and (x11))) or (((x10) and (x7)) or (x10))) or (x5)) ^ ((x9) or (x8))) ^ ((x10) and (x7))))) and (x11)) or (x0)) or (x13))) and (x11)) ^ ((x2) or (x4))) and ((x12) ^ (x11)))))) ^ ((x12) ^ (x11))) or (not (((x2) or (x4)) ^ (x4)))) ^ (x3)) ^ (x10)', 'classical_function': 'cf_v14_c24', 'python_function': 'pf_v14_c24'}, '(14,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '((((((((((((((((not ((((((not ((not (x13)) and (x3))) ^ (x12)) or (x0)) or (x8)) and ((x13) or (x12))) ^ (x4))) and (x2)) and (x2)) and (x12)) ^ ((x5) and ((x3) and (x7)))) ^ (x11)) or (not (x6))) and (x9)) ^ ((x5) and ((x3) and (x7)))) or (x4)) ^ (x2)) ^ (x0)) or ((x13) ^ ((x5) ^ ((x2) ^ (x4))))) ^ (x10)) and (not (x12))) or (not (x6))) ^ (((x5) ^ ((x2) ^ (x4))) ^ (x12))', 'classical_function': 'cf_v14_c25', 'python_function': 'pf_v14_c25'}, '(14,26)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': 'not ((((((((((((((not (((((not ((((((not (x4)) or (x2)) or (x3)) and ((x7) or (x8))) ^ (x1)) and (x0))) ^ (x6)) ^ (x4)) ^ (x12)) ^ (x0))) or (x10)) or (x8)) ^ (x2)) ^ (((x7) or (x8)) ^ ((x3) or (x5)))) ^ ((x13) or (not (((not (x8)) and ((x3) or (x5))) ^ ((not (x8)) and ((x3) or (x5))))))) ^ (x3)) and (not (x8))) or ((x7) or (x8))) and ((x2) ^ (x13))) ^ ((x0) ^ (x11))) ^ ((x9) or ((x0) and (x13)))) ^ (x3)) ^ (x13))', 'classical_function': 'cf_v14_c26', 'python_function': 'pf_v14_c26'}, '(14,27)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '((not (not (not ((not ((((((not (((not (((((((not (((((x13) ^ (x4)) ^ (x11)) ^ (x7)) or (x8))) ^ (x8)) and (x0)) or ((x9) ^ (x2))) ^ (x0)) ^ (x12)) and ((x9) or (x10)))) and (x3)) ^ (x6))) ^ (x5)) and (not (((x9) ^ (x2)) and (x8)))) ^ (not (((x9) ^ (x2)) and (x8)))) or ((x4) and (x4))) or ((x7) ^ ((x7) ^ (x12))))) ^ (x9))))) and (x5)) and ((x3) or ((x2) ^ (x1)))', 'classical_function': 'cf_v14_c27', 'python_function': 'pf_v14_c27'}, '(14,28)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'], 'statement': '(not (((((((((((((((((not (((not (not ((((((x10) or (x1)) or (x0)) or (x3)) or (x12)) or (x12)))) ^ ((x5) or ((x11) or (x2)))) ^ (x10))) or (((x6) or (x3)) or (x11))) and (not (x1))) or ((not ((x7) ^ (x7))) and (not (x1)))) ^ (x9)) ^ (x4)) or (not (x13))) or (x0)) and (x11)) or ((not (x13)) or (x7))) or ((x6) or (x3))) or ((x8) or (x1))) ^ (x7)) and (x2)) and (not (x1))) ^ ((x4) or (x4))) or (x0))) and ((x6) or (x3))', 'classical_function': 'cf_v14_c28', 'python_function': 'pf_v14_c28'}, '(15,19)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '((((not (((((((not ((((((((x5) and (x10)) or (x8)) and (x9)) ^ ((x3) or (x2))) ^ (x13)) or ((x11) and (x12))) ^ (x14))) and (x10)) or ((x13) or (x1))) ^ (x7)) ^ ((x13) ^ (x0))) ^ ((x13) ^ (x0))) and ((x13) ^ (x0)))) or (x4)) or ((x4) ^ (x7))) and ((x6) or (x12))) or (x11)', 'classical_function': 'cf_v15_c19', 'python_function': 'pf_v15_c19'}, '(15,22)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '(not (((((((((not (not (not (((((((not (((x10) ^ (x10)) and ((x12) ^ (x9)))) or (x13)) and (x7)) ^ (not (x3))) and (x6)) and (x8)) and (x6))))) and ((not (x3)) and (((x2) and (x11)) and (x5)))) ^ ((x2) and (x11))) or ((x2) and (x11))) and (x14)) or (not (x0))) ^ (not (x11))) ^ (x4)) and (x9))) and (not (x8))', 'classical_function': 'cf_v15_c22', 'python_function': 'pf_v15_c22'}, '(15,26)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '(((((((not ((not (not (not ((((((((((not ((not (((x7) or (x6)) and (x3))) or (x5))) and (x14)) or (x5)) and (x11)) or (x13)) or (x10)) or ((x9) or (x5))) ^ (x8)) ^ ((x2) or ((x11) or (x1)))) and (x0))))) and (x1))) or (x10)) ^ (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) ^ (x0)) and (x3)) or (not (not ((x4) and ((x2) ^ ((x12) and (x6))))))) and (((not ((x12) and (x14))) and (not ((x12) and (x14)))) or (x10))) and (((x2) ^ ((x12) and (x6))) or ((x11) or (x1)))', 'classical_function': 'cf_v15_c26', 'python_function': 'pf_v15_c26'}, '(15,27)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '((((not (((not ((not ((((not ((((not (((((not ((not (((x8) or (x10)) or (x3))) or (x5))) ^ (x10)) or (not (x6))) and ((x12) or ((x2) or (x14)))) and ((x9) and (x9)))) ^ (not (x14))) or (x2)) or (x8))) and ((x7) or (x8))) ^ (x1)) ^ ((x13) ^ (x14)))) and ((x12) or ((x2) or (x14))))) and ((x5) or (x11))) and (x4))) ^ (x0)) ^ (x0)) ^ ((x4) ^ (not (x6)))) ^ ((x3) or ((x12) or ((x2) or (x14))))', 'classical_function': 'cf_v15_c27', 'python_function': 'pf_v15_c27'}, '(15,28)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '(((not ((((not (((((not ((((((((not (not (((((not ((x3) ^ (x2))) ^ (x9)) and (x4)) and (x9)) and (x0)))) ^ (x5)) and (x0)) ^ ((x3) and (x7))) and (x1)) or (x13)) or ((x5) ^ (x0))) ^ ((not (x4)) and (x8)))) or (x10)) and ((x6) and (x4))) and ((x9) ^ (x12))) and ((not (x0)) and (x1)))) and (not (x4))) ^ (x7)) and (x7))) and (not (x11))) and (x0)) ^ (x14)', 'classical_function': 'cf_v15_c28', 'python_function': 'pf_v15_c28'}, '(15,29)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'], 'statement': '((((not (not ((((((((not ((not ((((not ((((((((((x5) ^ (x14)) and (x6)) ^ (x7)) or (x9)) ^ (x8)) and (x13)) and ((x5) or (x9))) and ((x5) or (x9))) ^ (x11))) ^ ((x10) ^ (x7))) ^ (x6)) ^ ((x5) or (x9)))) ^ (not (x0)))) and ((x5) or (x9))) and ((x12) ^ ((x5) or (x9)))) ^ ((x9) ^ ((x2) ^ (x11)))) ^ (x4)) or (not (x2))) or ((x8) and ((x2) ^ (x11)))) or (((x14) ^ (x3)) ^ (not (x0)))))) or (x5)) ^ (x13)) or ((x1) ^ (not (x10)))) or ((x12) ^ ((x5) or (x9)))', 'classical_function': 'cf_v15_c29', 'python_function': 'pf_v15_c29'}, '(16,24)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '((((((((((((((((((((((((x13) ^ (x8)) and (x13)) and (x11)) and (x2)) ^ (x12)) and (x2)) ^ ((x3) and (x9))) or ((x7) and ((x10) and (x14)))) or ((x13) ^ (x9))) and (x10)) or (x6)) and (x8)) or (x6)) or (x1)) and (x14)) ^ (x12)) or (((x15) ^ ((x13) ^ (x9))) and (x5))) ^ (x7)) ^ (x3)) and (x0)) and (x7)) ^ ((((x15) ^ ((x13) ^ (x9))) and (x5)) ^ (((x15) ^ ((x13) ^ (x9))) and (x5)))) ^ (x3)) and ((x4) ^ ((x2) or (x15)))', 'classical_function': 'cf_v16_c24', 'python_function': 'pf_v16_c24'}, '(16,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '(((((((((((((((not ((((not ((not ((((x5) or (x14)) and ((x4) or (x4))) and (x2))) ^ (x12))) or (x14)) and ((x3) and (x4))) and (x0))) ^ (x8)) ^ (x14)) ^ (x6)) or ((x15) or ((x2) ^ (x8)))) and (x15)) ^ ((x9) ^ (x10))) ^ (x14)) and ((x4) or (x4))) and (not (x13))) or (x3)) ^ ((((x7) and ((x4) or (x4))) or ((x4) or (x4))) ^ (x4))) ^ (x15)) ^ ((x9) ^ (x10))) or (x11)) or ((x3) and (((x15) or ((x2) ^ (x8))) ^ ((x7) and ((x4) or (x4)))))', 'classical_function': 'cf_v16_c25', 'python_function': 'pf_v16_c25'}, '(16,26)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '((not (((((((not (not ((((((((((((((not (not (x15))) and (x9)) ^ (x3)) or (x9)) and ((x4) or (x15))) or (x12)) or (x2)) and (x7)) and (x0)) or ((x0) or (not (x4)))) or ((x0) or (not (x4)))) or ((x1) ^ (x0))) or (x10)) ^ ((x4) or (x13))))) ^ (x4)) and ((not (x4)) and (x13))) and (x11)) and (((x8) or (x1)) or (x9))) and (x14)) and (not (x13)))) and ((x10) and (x6))) ^ ((x5) and ((x1) ^ (x0)))', 'classical_function': 'cf_v16_c26', 'python_function': 'pf_v16_c26'}, '(16,27)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '((((((((((((((not (not (((((not ((((not (((x4) or (x7)) ^ (x2))) and (x8)) and (x7)) and (x7))) or (x14)) ^ (x12)) and (x13)) and (((x12) and (x15)) ^ (x0))))) or (x12)) ^ (((x15) ^ (x10)) and (not (x4)))) and (x2)) and (x8)) or (x10)) and ((x9) and (x5))) or (x15)) and (not (x12))) or (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (((x5) and (x15)) or (x2))) and (((x10) and ((x9) and (x5))) and (((x12) and (x6)) or (x3)))) and (x11)) or ((x12) and (x6))) ^ (x5)', 'classical_function': 'cf_v16_c27', 'python_function': 'pf_v16_c27'}, '(16,28)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '(not (not ((not ((((((not ((((not ((not ((((((not (((((not (x9)) and (x14)) and (x8)) ^ (x2)) ^ (x5))) and (x13)) or (x1)) and (not (not (x3)))) ^ (x13)) ^ (x5))) ^ ((x15) or (x6)))) and (x11)) and (x9)) or (x10))) and ((x6) ^ (x4))) or (x11)) and (x12)) and ((x2) and (((x3) and (x0)) ^ (x11)))) ^ (((x15) and (x1)) and ((x7) ^ (x8))))) and ((x15) or (((x7) ^ (x8)) ^ (x5)))))) or ((x5) and ((x15) and (x1)))', 'classical_function': 'cf_v16_c28', 'python_function': 'pf_v16_c28'}, '(16,29)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': '(((not (not ((((((not (((((((((not (((not (((((((x6) ^ (x1)) ^ (x15)) and (x6)) and (x3)) and (x3)) ^ (x2))) and (x13)) ^ ((x13) and (x12)))) and ((x5) and ((x8) and ((x6) ^ (x10))))) and (x4)) or (x4)) ^ (x3)) and (x12)) ^ ((x9) ^ (x12))) and ((x3) and (x0))) or (x1))) and ((x3) ^ (x14))) and (x13)) or ((x15) or (x11))) and (x11)) or ((not ((x9) ^ (x12))) and (x14))))) or ((x7) or (x7))) or ((x6) ^ (x8))) and (x2)', 'classical_function': 'cf_v16_c29', 'python_function': 'pf_v16_c29'}, '(16,30)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'], 'statement': 'not (((((((not (not ((((not (((((((((((((((((not (x10)) and (x9)) ^ (x5)) ^ (not (x9))) and ((x0) and (x15))) and (x5)) ^ (x4)) ^ ((x7) ^ ((x0) and (x15)))) or ((x0) and (x15))) or ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x10)) or (x2)) ^ (x6)) and (x11)) ^ ((x2) and (x4))) or (x14)) and (x14))) or ((x9) and (x8))) ^ ((x1) ^ ((x7) ^ ((x0) and (x15))))) ^ (x8)))) and ((x3) ^ ((x12) ^ (x2)))) and (not ((x12) or (x4)))) or ((not (x9)) and (x8))) and (x2)) or (x13)) and (x13))', 'classical_function': 'cf_v16_c30', 'python_function': 'pf_v16_c30'}, '(17,25)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'], 'statement': '(not (((not (not ((((((((((((((((((((x2) ^ (x5)) and (x2)) or (x16)) or ((x15) and (x16))) and (x16)) or ((x12) and (x2))) ^ (x0)) ^ (x9)) or ((x1) or ((x12) and (x2)))) and (((x3) ^ (x6)) ^ ((x9) or (x10)))) or ((x15) and (x16))) and ((x13) ^ (x6))) and (x8)) and (((x9) or (x10)) or (x6))) and ((x4) or (x1))) and (x11)) ^ ((x15) and (x16))) and (x2)) ^ (x14)))) or (x13)) and ((x12) and (x2)))) or ((x2) or (x7))', 'classical_function': 'cf_v17_c25', 'python_function': 'pf_v17_c25'}, '(17,27)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'], 'statement': '(((((((((((not ((((((((((((((((x0) and (x9)) ^ (x9)) and (x7)) or (x4)) or (x4)) or (x15)) or ((x13) or (x4))) or (x3)) or (x2)) ^ (((x7) and (x14)) or (x14))) or (((x7) and (x14)) or (x8))) or (((x7) and (x14)) or (x8))) ^ (((x7) and (x14)) or (x14))) ^ ((x0) or (x6))) and (x11))) or (x15)) and (((x7) and (x14)) or (x8))) ^ (x10)) or (x6)) ^ ((x7) and (x14))) or (x7)) or ((x5) or (x12))) ^ (x14)) or (x13)) and (((x7) and (x14)) or (x14))) and ((x16) or (x15))', 'classical_function': 'cf_v17_c27', 'python_function': 'pf_v17_c27'}, '(17,30)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'], 'statement': '((((not (((((((not (((((((((((((((((((x6) or (x15)) ^ (x7)) or (x10)) and (x7)) ^ (x14)) ^ (x0)) and (x12)) or (x16)) or (x4)) ^ ((x2) ^ (x5))) or (x4)) or (x6)) and (x11)) or ((x7) or (x14))) or (x15)) ^ ((x8) or (x14))) and ((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13)))) ^ (((x8) or (x14)) and (x0)))) and ((x2) ^ (x5))) or (x15)) or ((x0) and (x14))) ^ (not (((x8) or (x14)) and (x0)))) and ((((((x7) or (x14)) or (x13)) or (((x7) or (x14)) or (x13))) ^ (x3)) or ((x12) or ((x3) and (x2))))) or ((x12) or ((x3) and (x2))))) ^ ((x15) and (not (((x8) or (x14)) and (x0))))) or (((x7) or (x14)) or (x13))) or (x2)) ^ (x9)', 'classical_function': 'cf_v17_c30', 'python_function': 'pf_v17_c30'}, '(17,31)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'], 'statement': 'not (((((((((((not (((not (((((not ((not (((((not (((not ((x2) and (x0))) or (x9)) or (x12))) or (x9)) and (x6)) ^ (x7)) or (x4))) and (x16))) or (not ((not (x1)) and (not (x1))))) and ((x13) or (x1))) ^ (x15)) and (x10))) and ((x3) and (x15))) ^ (((x16) or (x14)) and ((not (x1)) and (not (x1)))))) and ((x1) and ((x0) and (x16)))) and ((x0) and (x16))) or ((x8) or (x6))) and (x11)) ^ (not ((x0) ^ ((x16) or (x14))))) or ((not (not (x10))) ^ ((x0) ^ ((x16) or (x14))))) and (x14)) and (x5)) and ((x8) and (x2))) ^ ((x1) and ((x0) and (x16))))', 'classical_function': 'cf_v17_c31', 'python_function': 'pf_v17_c31'}, '(18,30)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17'], 'statement': '((((((((((((((not (((not (((not ((not ((((((not ((x4) or (x14))) and (x3)) ^ (x3)) ^ (x13)) or (x3)) ^ (x10))) ^ ((x17) or (x11)))) and (x17)) ^ (x9))) or (((x11) ^ (x15)) ^ (x15))) and ((x6) ^ (x12)))) or (x11)) ^ (x16)) and (not (x17))) ^ (x3)) or (x8)) or (x5)) or ((x15) or (((x11) ^ (x15)) ^ (x15)))) ^ ((x14) ^ (x8))) or (x0)) and ((x11) ^ (x15))) or ((x4) ^ (x7))) or (x2)) or (not (x9))) or (((x14) ^ (x8)) or ((x11) ^ (x15)))', 'classical_function': 'cf_v18_c30', 'python_function': 'pf_v18_c30'}, '(18,31)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17'], 'statement': '(((not (not (not ((not (((((((not (((((not ((not ((((((((((x14) or (x16)) and (x4)) and (x11)) or (x9)) ^ (x4)) ^ (x2)) or ((x9) or (x13))) and (x3)) ^ ((x12) or (x3)))) and (x15))) ^ (x5)) ^ (((x1) ^ (x5)) ^ ((x14) ^ (x15)))) ^ ((x1) ^ (x5))) and (x8))) and (x10)) or (x7)) or (((x12) and (x16)) and (x17))) and (((x13) or (x5)) or ((x7) or (x0)))) and ((x6) and (x16))) or (x6))) and ((x9) or (x13)))))) and (x13)) or (not ((x10) and ((x14) ^ (x15))))) or (x3)', 'classical_function': 'cf_v18_c31', 'python_function': 'pf_v18_c31'}, '(19,32)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18'], 'statement': '((((not ((not (((((((((((((((((((not ((not ((not ((not (x16)) and (x2))) or ((x0) and (x13)))) or (x2))) ^ (x18)) and (x6)) or (((x0) and (x13)) ^ (x9))) ^ ((x0) and (x13))) ^ (x0)) ^ (x4)) and (not ((x0) or (x1)))) and (x11)) or ((not (not (x7))) and (x5))) or (x12)) and (((x0) or (x1)) or (x8))) and (x4)) or (not (x7))) or (((x0) or (x1)) or (x8))) and (x17)) ^ ((((x0) or (x1)) or (x8)) ^ (x3))) ^ (x15)) and ((x8) or (x6)))) ^ ((x16) and (x10)))) ^ (x14)) or (((x0) and (x13)) or (x6))) ^ (x0)) ^ ((not (not (x7))) and (x5))', 'classical_function': 'cf_v19_c32', 'python_function': 'pf_v19_c32'}, '(19,33)': {'variables': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18'], 'statement': '(((((not ((not ((((((((((((((((((((((not (not (not (not (x0))))) and (not (x10))) or ((x16) and (not (x10)))) ^ (x12)) or (x17)) or (x12)) or (x8)) and (x9)) and (x3)) ^ (x14)) or ((x16) and (not (x10)))) and ((x7) or (x16))) or ((x1) and (x15))) ^ (((x10) or (not (x10))) and (x15))) or (x6)) and (x2)) or ((x17) or (x11))) or (x13)) ^ (x4)) ^ (((x10) or (not (x10))) and (x15))) or ((x18) or (x13))) and ((x16) and (not (x10))))) ^ (((((x17) or (x11)) ^ (x7)) or (not (x5))) or (((x1) and (x15)) ^ (x15))))) ^ (x14)) and (x9)) or (not (x5))) or ((x18) or (x13))) or (not (x5))', 'classical_function': 'cf_v19_c33', 'python_function': 'pf_v19_c33'}}