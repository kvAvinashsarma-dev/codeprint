scores=[86, 65, 53, 100, 40, 67, 97, 50]   
sum1=0
for x in scores:
    sum1 = sum1+x
avg = sum1/len(scores)
print("average is", avg)
if avg >= 50:  
    print("pass")
# debug
else:
    print("fail")
