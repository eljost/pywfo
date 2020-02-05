from SD import SD


length = 6

sd37 = SD(3, 7, length, "bra")
print(sd37)

sd48 = SD(4, 8, length, "ket")
print(sd48)

print("sd37 == sd48", sd37 == sd48)
