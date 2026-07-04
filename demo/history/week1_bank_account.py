class Account:
    def __init__(self):
        self.b = 0
    def deposit(self, amt):
# copied from my notes
        self.b = self.b+amt
    def withdraw(self, amt):
        if amt>self.b:
# fixme later

            print('not enough money')

        else:
            self.b =self.b-amt 

# not sure if this works
acc=Account()
acc.deposit(393)
acc.withdraw(40)
print(acc.b)
# print(ans)
