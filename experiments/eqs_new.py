import math

def log_n_and(A,B,N):
    # Does the pairwise xor, zip in Haskell speak 
    res = [not (a^b) for (a,b) in zip(A,B)]
    # Prints the actual op that needs to be done for tweedledum
    for c in range(N):
        print("res_"+str(N)+"_"+str(c)+" = not (A["+str(c)+"] ^ B["+str(c)+"])")
    i = N
    # Outer loop does ANDs on each pair of adjacent values in res, copies result in new res 
    while i > 1:
        res_new = []
        j = 0
        while j < i-1:
            res_new.append(res[j]&res[j+1])
            print("res_"+str(math.ceil(i/2))+"_"+str(math.ceil(j/2))+" = "+"res_"+str(i)+"_"+str(j)+" & "+"res_"+str(i)+"_"+str(j+1))
            j = j+2
        if i%2 != 0:
            res_new.append(res[j]) # copies the odd one down                                                                                                                                                                                 
            print("res_"+str(math.ceil(i/2))+"_"+str(math.ceil(j/2))+" = res_"+str(i)+"_"+str(j))
        i = math.ceil(i/2)
        res = res_new
    return res[0]


def seq_and(A,B,N):
    nxor = [not (a^b) for (a,b) in zip(A,B)]
    for c in range(N):
        print("nxor_"+str(c)+" = not (A["+str(c)+"] ^ B["+str(c)+"])")
    res = nxor[0]
    print("res_0 = nxor_0")
    for i in range(1,N):
        res = res & nxor[i]
        print("res_"+str(i)+" = res_"+str(i-1)+" & nxor_"+str(i))
    return res    

A = [1,0,0,1,0,0,1,0,1,0,1]#,1,1,1]
B = [1,0,0,1,0,0,1,0,1,1,1]#,1,1,1]
N = 11

result = log_n_and(A,B,N)
print(A, " == ", B, " is ", result)

print("\n\n")
result_seq = seq_and(A,B,N)
print(A, " == ", B, " is ", result_seq)

