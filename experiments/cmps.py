def log_n_cmp(A,B,N):
    #flag[i] = 1 means i-th bit gives an answer 
    #initialize flag and result lists
    flag = [a^b for (a,b) in zip(A,B)]
    res = []
    #res[i] is B[i] if bits are different; 0 otherwise
    for c in range(N):
        res.append(flag[c] & B[c])
    #printing initialization for unrolled/linear circuit     
    for c in range(N):
        print("flag_0_"+str(c)+" = A["+str(c)+"] ^ B["+str(c)+"]")
        print("res_0_"+str(c)+" = flag_0_"+str(c)+" & B["+str(c)+"]")
    step = 1
    while step <= N/2:
        i = 0
        while i < N:
            #if flag[i] is one, then res[i] holds comparison result
            #res[i] = MUX(flag[i],res[i],res[i+step])
            #if flag[i] is one, then keep the flag value, otherwise set to flag of i+step
            #flag[i] = MUX(flag[i],flag[i],flag[i+step]
            res[i] = (flag[i] & res[i]) ^ ((not flag[i]) & res[i+step])
            flag[i] = flag[i] ^ ((not flag[i]) & flag[i+step])
            print("t1_"+str(step)+"_"+str(i)+" = flag_"+str(step//2)+"_"+str(i)+" & res_"+str(step//2)+"_"+str(i))
            print("t2_"+str(step)+"_"+str(i)+" = (not flag_"+str(step//2)+"_"+str(i)+") & res_"+str(step//2)+"_"+str(i+step))
            print("res_"+str(step)+"_"+str(i)+" = t1_"+str(step)+"_"+str(i)+" ^ t2_"+str(step)+"_"+str(i))
            print("t4_"+str(step)+"_"+str(i)+" = (not flag_"+str(step//2)+"_"+str(i)+") & flag_"+str(step//2)+"_"+str(i+step))
            print("flag_"+str(step)+"_"+str(i)+" = flag_"+str(step//2)+"_"+str(i)+" ^ t4_"+str(step)+"_"+str(i))
            i = i + step*2
        step = step*2
    print("return res_"+str(int(N/2))+"_0")
    return res[0]    


def seq_cmp(A,B,N):
    xor = []
    for c in range(N):
        xor.append(A[c]^B[c])
        print("xor_"+str(c)+" = A["+str(c)+"] ^ B["+str(c)+"]")
    res = False
    flag = False
    print("res_0 = False")
    print("flag_0 = False")
    for i in range(0,N):
        c = xor[i] & (not flag)
        print("c_"+str(i)+" = xor_"+str(i)+" & (not flag_"+str(i)+")")
        flag =(c & True) ^ ((not c) & flag)
        print("t2_"+str(i)+" = (not c_"+str(i)+") & flag_"+str(i))
        print("flag_"+str(i+1)+" = c_"+str(i)+" ^ t2_"+str(i)) 
        res = (c & B[i]) ^ ((not c) & res) # res2 = MUX(c,B[i],res1)
        print("t3_"+str(i)+" = c_"+str(i)+" & B["+str(i)+"]")
        print("t4_"+str(i)+" = (not c_"+str(i)+") & res_"+str(i))
        print("res_"+str(i+1)+" = t3_"+str(i)+" ^ t4_"+str(i))
    print("return res_"+str(N))
    return res    

A = [1,1,0,1] #,0,0,1,1]
B = [1,0,1,1] #,0,1,1,1]
N = 4

result = log_n_cmp(A,B,N)
print(A, " < ", B, " is ", result)

print("\n\n")
result_seq = seq_cmp(A,B,N)
print(A, " < ", B, " is ", result_seq)


print("\n\n")
A = [1, 0]
B = [0, 0]
N = 2
result = seq_cmp(A,B,N)
print(A, " < ", B, " is ", result)

