import matplotlib.pyplot as plt

odd_numbers = []

with open('1.txt', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index % 2 == 0:
            continue
        values = line.split()
        last_value = values[-1]
        odd_numbers.append(float(last_value))

plt.plot(odd_numbers, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()