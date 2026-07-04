marks = [71, 86, 50, 59, 78, 77, 80, 45]  
tot= 0
# debug
for x in marks:
    tot= tot+x
avg= tot/len(marks)
print('average is', avg)
if avg>=60:
    print('pass')
else:
    print('fail')
