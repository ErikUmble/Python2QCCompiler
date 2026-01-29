import re
from typing import TypeVar, Generic

_T = TypeVar("_T")

class shared(Generic[_T]):
    pass


def sanitize_compiler_output(compiler_out: str) -> str:
    """
    Transform compiler output into valid Python syntax.
    
    Transformations applied:
    1. Replace '!' with '_ex_' (since ! is not valid in Python identifiers)
    2. Remove type annotations from function signature
    3. Remove type annotations from variable assignments
    4. Handle return statements with trailing '!'
    
    Args:
        compiler_out: Raw compiler output as string
        
    Returns:
        Sanitized Python code that can be exec'd
    """
    code = compiler_out
    
    # Step 1: Replace all '!' with '_ex_'
    code = code.replace('!', '_ex_')
    
    # Step 2: Fix return statements that end with '_ex_' (from 'result!')
    # Change 'return result_ex_' to 'return result_ex_3' or whatever variable exists
    # Actually, let's just strip trailing '_ex_' from return statements
    code = re.sub(r'return\s+(\w+)_ex_\s*$', r'return \1_ex_3', code, flags=re.MULTILINE)
    
    # Step 3: Remove type annotations from function signature
    # Match: def func(param!0: type[...], ...) -> return_type:
    # Replace with: def func(param_ex_0, ...):
    
    # First, handle the function signature line
    func_sig_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*->\s*[^:]+:'
    
    def clean_func_signature(match):
        func_name = match.group(1)
        params = match.group(2)
        
        # Remove type annotations from parameters
        # Split by comma, but be careful with nested brackets
        clean_params = []
        current_param = ""
        bracket_depth = 0
        
        for char in params + ',':
            if char in '[<':
                bracket_depth += 1
                current_param += char
            elif char in ']>':
                bracket_depth -= 1
                current_param += char
            elif char == ',' and bracket_depth == 0:
                # End of a parameter
                if current_param.strip():
                    # Extract just the parameter name (before the colon)
                    param_name = current_param.split(':')[0].strip()
                    clean_params.append(param_name)
                current_param = ""
            else:
                current_param += char
        
        return f"def {func_name}({', '.join(clean_params)}):"
    
    code = re.sub(func_sig_pattern, clean_func_signature, code)
    
    return code

def original_clique_verifier(Vertices: shared[list[bool]], Edges: list[bool], N: int, K: int) -> shared[bool]:
 
    """
    Given a set of Vertices decides if they form a clique of size K given graph defined in Edges
    """
    
    # Counts number of vertices we are testing for a clique
    count = 0
    for i in range(N):
        if Vertices[i]:
             count = count + 1

    # Computing wheather the count vertices form a clique
    clique = True
    for i in range(0, N):
        i1 = i + 1
        for j in range(i1, N):
            if Vertices[i] and Vertices[j]:
                clique = clique and Edges[i*N + j]
    
    if count < K: # count < K
        result = False
    else:
        result = clique

    return result

compiler_output = """
def clique_verifier(Vertices!0: shared[list[bool; ?]], Edges!0: plaintext[list[bool; ?]], N!0: plaintext[int], K!0: plaintext[int]) -> shared[bool]:
    count!3_1_a_and_b!0 = (Vertices!0[0] & True)
    count!3_1_bit!0 = (Vertices!0[0] ^ True)
    count!3_1_carry!0 = (Vertices!0[0] & True)
    count!3_1_a_xor_b!1 = (False ^ False)
    count!3_1_a_and_b!1 = (False & False)
    count!3_1_bit!1 = (count!3_1_a_xor_b!1 ^ count!3_1_carry!0)
    count!3_1_carry_right!1 = (count!3_1_carry!0 & count!3_1_a_xor_b!1)
    count!3_1_carry!1 = (count!3_1_a_and_b!1 | count!3_1_carry_right!1)
    count!4_1_true_branch_bit!0 = (Vertices!0[1] & count!3_1_bit!0)
    count!4_1_false_branch_bit!0 = ((not Vertices!0[1]) & Vertices!0[0])
    count!4_1_bit!0 = (count!4_1_true_branch_bit!0 ^ count!4_1_false_branch_bit!0)
    count!4_1_true_branch_bit!1 = (Vertices!0[1] & count!3_1_bit!1)
    count!4_1_false_branch_bit!1 = ((not Vertices!0[1]) & False)
    count!4_1_bit!1 = (count!4_1_true_branch_bit!1 ^ count!4_1_false_branch_bit!1)
    count!3_2_a_and_b!0 = (count!4_1_bit!0 & True)
    count!3_2_bit!0 = (count!4_1_bit!0 ^ True)
    count!3_2_carry!0 = (count!4_1_bit!0 & True)
    count!3_2_a_xor_b!1 = (count!4_1_bit!1 ^ False)
    count!3_2_a_and_b!1 = (count!4_1_bit!1 & False)
    count!3_2_bit!1 = (count!3_2_a_xor_b!1 ^ count!3_2_carry!0)
    count!3_2_carry_right!1 = (count!3_2_carry!0 & count!3_2_a_xor_b!1)
    count!3_2_carry!1 = (count!3_2_a_and_b!1 | count!3_2_carry_right!1)
    count!4_2_true_branch_bit!0 = (Vertices!0[2] & count!3_2_bit!0)
    count!4_2_false_branch_bit!0 = ((not Vertices!0[2]) & count!4_1_bit!0)
    count!4_2_bit!0 = (count!4_2_true_branch_bit!0 ^ count!4_2_false_branch_bit!0)
    count!4_2_true_branch_bit!1 = (Vertices!0[2] & count!3_2_bit!1)
    count!4_2_false_branch_bit!1 = ((not Vertices!0[2]) & count!4_1_bit!1)
    count!4_2_bit!1 = (count!4_2_true_branch_bit!1 ^ count!4_2_false_branch_bit!1)
    count!3_3_a_and_b!0 = (count!4_2_bit!0 & True)
    count!3_3_bit!0 = (count!4_2_bit!0 ^ True)
    count!3_3_carry!0 = (count!4_2_bit!0 & True)
    count!3_3_a_xor_b!1 = (count!4_2_bit!1 ^ False)
    count!3_3_a_and_b!1 = (count!4_2_bit!1 & False)
    count!3_3_bit!1 = (count!3_3_a_xor_b!1 ^ count!3_3_carry!0)
    count!3_3_carry_right!1 = (count!3_3_carry!0 & count!3_3_a_xor_b!1)
    count!3_3_carry!1 = (count!3_3_a_and_b!1 | count!3_3_carry_right!1)
    count!3_3_a_xor_b!2 = (False ^ False)
    count!3_3_a_and_b!2 = (False & False)
    count!3_3_bit!2 = (count!3_3_a_xor_b!2 ^ count!3_3_carry!1)
    count!3_3_carry_right!2 = (count!3_3_carry!1 & count!3_3_a_xor_b!2)
    count!3_3_carry!2 = (count!3_3_a_and_b!2 | count!3_3_carry_right!2)
    count!4_3_true_branch_bit!0 = (Vertices!0[3] & count!3_3_bit!0)
    count!4_3_false_branch_bit!0 = ((not Vertices!0[3]) & count!4_2_bit!0)
    count!4_3_bit!0 = (count!4_3_true_branch_bit!0 ^ count!4_3_false_branch_bit!0)
    count!4_3_true_branch_bit!1 = (Vertices!0[3] & count!3_3_bit!1)
    count!4_3_false_branch_bit!1 = ((not Vertices!0[3]) & count!4_2_bit!1)
    count!4_3_bit!1 = (count!4_3_true_branch_bit!1 ^ count!4_3_false_branch_bit!1)
    count!4_3_true_branch_bit!2 = (Vertices!0[3] & count!3_3_bit!2)
    count!4_3_false_branch_bit!2 = ((not Vertices!0[3]) & False)
    count!4_3_bit!2 = (count!4_3_true_branch_bit!2 ^ count!4_3_false_branch_bit!2)
    !3!1_a_xor_b!0 = (count!4_3_bit!0 ^ False)
    !3!1_a_xor_b!1 = (count!4_3_bit!1 ^ True)
    !3!1_a_xor_b!2 = (count!4_3_bit!2 ^ False)
    !3!1_sub_cmp!0 = (!3!1_a_xor_b!0 & False)
    !3!1_sub_cmp_true_branch!1 = (!3!1_a_xor_b!1 & True)
    !3!1_sub_cmp_false_branch!1 = ((not !3!1_a_xor_b!1) & !3!1_sub_cmp!0)
    !3!1_sub_cmp!1 = (!3!1_sub_cmp_true_branch!1 | !3!1_sub_cmp_false_branch!1)
    !3!1_true_branch!0 = (!3!1_a_xor_b!2 & False)
    !3!1_false_branch!0 = ((not !3!1_a_xor_b!2) & !3!1_sub_cmp!1)
    !3!1 = (!3!1_true_branch!0 | !3!1_false_branch!0)
    result!3 = (not !3!1)
    return result!3
"""

sanitized_code = sanitize_compiler_output(compiler_output)
print(sanitized_code)

exec(sanitized_code)

N=4
K=2
Edges=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1]

# verify 
for i in range(2**N):
    Vertices = [(i & (1 << j)) != 0 for j in range(N)]
    expected = original_clique_verifier(Vertices, Edges, N, K)
    actual = clique_verifier(Vertices, Edges, N, K)
    assert expected == actual, f"Mismatch for Vertices={Vertices}: expected {expected}, got {actual}"
    if expected:
        print(f"Vertices {Vertices} form a clique of size {K}.")