scores= [45, 57, 53, 65, 98, 40, 100]
s = 0
for x in scores:
    s = s+x

avg=s/len(scores)
# hw3
print("average is", avg)
if avg>=60:
    print("pass")
else:
    print("fail")
