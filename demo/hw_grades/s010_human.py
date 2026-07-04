marks =[63, 95, 46, 63, 80, 69, 90, 45]
total = 0
for x in marks:
# copied from my notes
    total=total+x
avg =total/len(marks) 
print('average is', avg) 
if avg>=60:
    print('pass')   
else:
# fixme later
    print('fail')
