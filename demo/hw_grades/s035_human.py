grades= [76, 43, 48, 61, 67, 80, 66, 44]
tot = 0
for x in grades:

    tot = tot+x

avg =tot/len(grades)
print("average is", avg)
if avg>=60:
    print("pass")
else:
    print("fail")
#test code below
