def log_n_add(A,B,N):
    res = [not (a^b) for (a,b) in zip(A,B)]
    for c in range(N):
        print("res_0_"+str(c)+" = not (A["+str(c)+"] ^ B["+str(c)+"])")
    step = 1
    while step <= N/2:
        i = 0
        while i < N:
            res[i] = res[i] & res[i+step]
            print("res_"+str(step)+"_"+str(i)+" = "+"res_"+str(step//2)+"_"+str(i)+" & "+"res_"+str(step//2)+"_"+str(i+step)) 
            i = i + step*2
        step = step*2
    return res[0]    

def seq_and(A,B,N):
    nxor = [not (a^b) for (a,b) in zip(A,B)]
    for c in range(N):
        #nxor.append(A[c]^B[c])
        print("nxor_"+str(c)+" = not (A["+str(c)+"] ^ B["+str(c)+"])")
    res = True
    print("res_0 = True")
    for i in range(1,N+1):
        res = res & nxor[i-1]
        print("res_"+str(i)+" = res_"+str(i-1)+" & nxor_"+str(i-1))
    return res    

A = [1,0,1,0,0,1,1,1]
B = [1,1,0,1,1,1,1,1]
N = 8

result = log_n_add(A,B,N)
print(A, " == ", B, " is ", result)

print("\n\n")
result_seq = seq_and(A,B,N)
print(A, " == ", B, " is ", result_seq)

