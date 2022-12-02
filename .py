a=float(input("enter the first side of triangle"))
b=float(input("enter the second side of triangle"))
c=float(input("enter the third side of triangle"))

s=(a+b+c)/2
area=(s*(s-a)*(s-b)*(s-c)) ** 0.5
print("the area of the triangle is:",area)