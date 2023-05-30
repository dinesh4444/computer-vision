import numpy as np 

# One dimensional array
arr = np.array([1,2,3,])
print(arr)

# Two dimensional array
arr1 = np.array([(1,2,3,),(4,5,6)])
print(arr1)

# Tree dimensional array (Basically used for 3 channel image)
arr2 = np.array([(1,2,3),(4,5,6)],[(1,2,3,),(4,5,6)],[(1,2,3),(4,5,6)])
print(arr2)

# creating 0 number 3X3 matrix or array
arr3 = np.zeros((3,3), dtype='int')
print(arr3)

# int: for 8 bit image range -128 to +128
# unit: for 8 bit image range 0 to 255

# creating 3x3 matrix with 1
arr4 = np.ones((3,3), dtype='int')
print(arr4)

# creating 3x3 matrix with specific number
arr5 = np.full((3,3), fill_value=5)
print(arr5)

# used for making kernal, mask to apply on images

# making identical matrix
arr6 = np.eye(3)
print(arr6)

arr7 = np.arange(1,10)
print(arr7)

# making array of same spacing values
arr8 = np.linspace(2,5,10)
print(arr8)

# making array with random number
arr9 = np.random.randint(15,50,9)
print(arr9)

# appending any number to array
arr11 = np.append([9,8,7])
