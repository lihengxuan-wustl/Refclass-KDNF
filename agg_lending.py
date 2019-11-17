print 'lending nn'
precision=0
coverage=0
count=0
for j in range(1133):
    if(j>=953 and j<=955):
	continue
    filename='out_pickles/lending-setcover-nn-'+str(j)+'.txt'
    with open(filename,'r') as f:
        lines=f.readlines()
        for i in range(len(lines)/3):
            precision+=float(lines[i*3+1])
            coverage += float(lines[i * 3 + 2])
            count+=1
print(precision/count)
print(coverage/count)

print 'lending xgboost'
precision=0
coverage=0
count=0
for j in range(1133):
    if(j>=953 and j<=955):
        continue
    filename='out_pickles/lending-setcover-xgboost-'+str(j)+'.txt'
    with open(filename,'r') as f:
        lines=f.readlines()
        for i in range(len(lines)/3):
            precision+=float(lines[i*3+1])
            coverage += float(lines[i * 3 + 2])
            count+=1
print(precision/count)
print(coverage/count)
print 'lending logistic'
precision=0
coverage=0
count=0
for j in range(1133):
    if(j>=953 and j<=955):
	continue
    filename='out_pickles/lending-setcover-logistic-'+str(j)+'.txt'
    with open(filename,'r') as f:
        lines=f.readlines()
        for i in range(len(lines)/3):
            precision+=float(lines[i*3+1])
            coverage += float(lines[i * 3 + 2])
            count+=1
print(precision/count)
print(coverage/count)


