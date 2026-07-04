# hw4 stats thing
nums= [23.5, 67.0, 45.2, 89.1, 12.8, 55.5]

total =0
for x in nums:
    total = total+x
avg= total/len(nums)
# print(total)

mn = nums[0]
mx=nums[0]
for x in nums:
    if x<mn:
        mn = x
    if x>mx:
        mx=x

print('mean', avg)
print("min", mn)
#print("test", nums)
print('max', mx)
# TODO make this a function maybe
