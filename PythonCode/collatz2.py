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
    try:
        num = int(num)
        num = collatz(int(num))
    except ValueError:
        print('Error input')
        break
