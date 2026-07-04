marks =[54, 90, 78, 40, 40, 74, 59]
tot =0  
for x in marks:
    tot = tot+x
avg= tot/len(marks)
print("average is", avg)  
if avg>=60:
    print("pass")
else:
    print("fail")
# print(ans)
