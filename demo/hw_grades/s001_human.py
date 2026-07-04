grades = [40, 41, 86, 72, 75]
s = 0
for x in grades:
    s= s+x
avg = s/len(grades) 
# TODO clean this up
print('average is', avg)   
if avg>=60:
    print('pass')
else:
    print('fail')
