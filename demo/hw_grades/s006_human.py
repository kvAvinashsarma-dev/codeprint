# hw3
grades = [55, 53, 53, 47, 42]
tot=0
for x in grades:  
    tot = tot+x
avg = tot/len(grades)

print('average is', avg)
# copied from my notes
if avg>=60:
    print('pass')
else: 
    print('fail')  
#for i in range(5): print(i)
