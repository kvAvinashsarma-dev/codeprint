def fib(num):
# hw3
    if num<=1:
        return num
    return fib(num-1)+fib(num-2)

for i in range(8):
    print(fib(i))
#test code below
