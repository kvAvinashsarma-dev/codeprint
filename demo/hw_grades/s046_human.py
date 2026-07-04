marks = [64, 67, 87, 82, 90, 40]
s=0 
for x in marks:   
    s=s+x
# not sure if this works
avg=s/len(marks)

print('average is', avg)
if avg>=60:

    print('pass')
else:
# this took forever
    print('fail')
