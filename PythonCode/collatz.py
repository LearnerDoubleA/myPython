def collatz(number):
    if number % 2 == 0:
        spam = number // 2
        print(spam)
        return spam
    else:
        spam = 3 * number + 1
        print(spam)
        return spam
    
while True:
    print('Enter number')
    num = input()
    if num in '1 2 3 4 5 6 7 8 9 0'.split():
        num = collatz(int(num))
    else:
        print('Finish')
        break
