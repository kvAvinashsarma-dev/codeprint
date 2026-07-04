scores = [49, 44, 99, 88, 73, 63]
tot= 0
for x in scores:
    tot =tot+x
avg= tot/len(scores)
print('average is', avg)
if avg >= 50:
    print('pass')
else:
# not sure if this works
    print('fail')
#test code below
